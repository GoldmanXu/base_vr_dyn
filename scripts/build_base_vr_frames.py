#!/usr/bin/env python
# Build the Base / VideoReward (/ Image) frame-strip demo.
#
# Selection rule (applied to EVERY section): keep a clip only if its VideoReward
# variant is dynamic, i.e. VBench dynamic_degree == 1 ("vr=true"). Nothing else is
# required of the other variants. In practice this only drops clips from Kling; the
# curated cog/ltx/hy sets are already all vr-dynamic, so they are unchanged.
#
# Per clip one composite (top -> bottom): Base, VideoReward, and -- for Kling only --
# Image (HPSv3). Four frames are sampled at equal time points (start / 1/3 / 2/3 / end).
# Frames are pasted at native resolution (capped at MAX_TW=480 wide) and saved JPEG q92.
import json, os, glob, html, shutil, subprocess
from PIL import Image, ImageDraw, ImageFont

VB     = "/data_2/xujinyuan/vbench_check"
ROOT   = VB + "/base0_vr1_videos"            # ltx/hy single-dir backbones
PDIR   = os.path.join(ROOT, "propmt")
WORK   = VB + "/base_vr_dyn_repo"
OUTDIR = os.path.join(WORK, "frames")

# CogVideoX (check_v5): all 6 seeds, base/vr pairs live in per-seed dirs; prompts + flips
# in the by-seed eval JSON; per-clip vr dynamic_degree from the per-seed VBench results.
COG_VID_ROOT = VB + "/eval_check_v5_cogvideox/base0_vr1_videos"          # seed_<S>/
COG_JSON     = VB + "/eval_check_v5_cogvideox/base0_vr1_by_seed.json"
COG_RES      = VB + "/eval_check_v5_cogvideox/results/seed_%s/ly_video_vr_only/*eval_results.json"

# LTX-Video (seed 3) and HunyuanVideo (seed 42): the demo dirs hold the curated dyn_NNN
# pairs; per-clip vr dynamic_degree comes from the full VBench grid keyed dyn_NNN-<seed>.
LTX_RES, LTX_SEED = VB + "/eval_check_v5_ltx/results/ly_video_vr_only/dyn_full/*eval_results.json", "3"
HY_RES,  HY_SEED  = [VB + "/eval_hy/results/ly_video_vr_only/dyn_degree/*eval_results.json",
                     VB + "/eval_hy/results/ly_video_vr_only/dyn_degree_extra/*eval_results.json"], "42"

# Kling v3 Pro (check_v4 kling_batch): variants are top-level dirs; one mp4 per clip name.
KLING_VID     = "/data_2/xujinyuan/diff_check/check_v4/kling_batch/videos"
KLING_BASE    = "base"
KLING_VR      = "ly_video_vr_only"
KLING_IMAGE   = "hk_uni_hspv3_image"
KLING_PROMPTS = VB + "/eval_kling/prompts/ly_video_vr_only.json"          # "<name>.mp4" -> prompt
KLING_RES     = VB + "/eval_kling/results/ly_video_vr_only/*eval_results.json"

FRACS  = [0.0, 1.0/3.0, 2.0/3.0, 1.0]                  # equal time points
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
JPEG_Q = 92
MAX_TW = 480                                           # cap per-frame width (cog/ltx/hy<=480 -> native)

# reference geometry at a 200px frame width; scaled to render width so proportions hold.
TW_REF = 200
G0 = dict(GAPF=4, LBL=150, LBLGAP=8, HDR=30, GAPR=8, EDGE=10, FH=18, FL=19)


def _out(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()

def nframes(path):
    return int(_out("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", path))

def dims(path):
    w, h = _out("ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path).split("x")
    return int(w), int(h)

def idxs(n):
    return [int(round(f * (n - 1))) for f in FRACS]

def vr_dynamic(pat):
    """basename(no ext) -> bool, from a VBench eval_results.json dynamic_degree block."""
    d = json.load(open(sorted(glob.glob(pat))[0]))
    _, lst = d["dynamic_degree"]
    return {os.path.basename(e["video_path"])[:-4]: bool(e["video_results"]) for e in lst}

def extract(video, indices, tmp):
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    sel = "+".join("eq(n\\,{})".format(i) for i in indices)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video,
        "-vf", "select={}".format(sel), "-vsync", "0",
        os.path.join(tmp, "f_%03d.png")], check=True)
    return [os.path.join(tmp, "f_{:03d}.png".format(k)) for k in range(1, len(indices) + 1)]

def ctext(draw, cx, cy, txt, font, fill=(0, 0, 0)):
    b = draw.textbbox((0, 0), txt, font=font)
    draw.text((cx - (b[2]-b[0])/2, cy - (b[3]-b[1])/2 - b[1]), txt, font=font, fill=fill)

def fit_font(label, base_size, maxw):
    """Largest font <= base_size whose label width fits maxw (no-op for short labels)."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = base_size
    while size > 8:
        f = ImageFont.truetype(FONT_B, size)
        b = probe.textbbox((0, 0), label, font=f)
        if (b[2] - b[0]) <= maxw:
            return f
        size -= 1
    return ImageFont.truetype(FONT_B, size)

def geom_for(w):
    s = w / float(TW_REF)
    return {k: max(1, int(round(v * s))) for k, v in G0.items()}

def build_strip(rows, w, h, g, out_jpg, tmptag):
    """rows: list of (label, video_path). Header (frame indices) taken from the first row."""
    TW, TH = w, h
    f_hdr = ImageFont.truetype(FONT_B, g["FH"])
    nrows = len(rows)
    frames_w = 4 * TW + 3 * g["GAPF"]
    GW = g["EDGE"] + g["LBL"] + g["LBLGAP"] + frames_w + g["EDGE"]
    GH = g["EDGE"] + g["HDR"] + nrows * TH + (nrows - 1) * g["GAPR"] + g["EDGE"]
    canvas = Image.new("RGB", (GW, GH), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    frames_x0 = g["EDGE"] + g["LBL"] + g["LBLGAP"]
    header_idx = None
    for r, (label, vid) in enumerate(rows):
        ind = idxs(nframes(vid))
        if header_idx is None:
            header_idx = ind
            for c, fi in enumerate(ind):
                cx = frames_x0 + c * (TW + g["GAPF"]) + TW / 2
                ctext(draw, cx, g["EDGE"] + g["HDR"] / 2, "frame {}".format(fi), f_hdr)
        tmp = "/tmp/bvf_{}_{}".format(tmptag, r)
        pngs = extract(vid, ind, tmp)
        y = g["EDGE"] + g["HDR"] + r * (TH + g["GAPR"])
        ctext(draw, g["EDGE"] + g["LBL"] / 2, y + TH / 2, label, fit_font(label, g["FL"], g["LBL"]))
        for c, p in enumerate(pngs):
            im = Image.open(p).convert("RGB")
            if im.size != (TW, TH):
                im = im.resize((TW, TH), Image.LANCZOS)
            canvas.paste(im, (int(frames_x0 + c * (TW + g["GAPF"])), int(y)))
        shutil.rmtree(tmp)
    canvas.save(out_jpg, quality=JPEG_Q, optimize=True, progressive=True)
    return GW, GH, header_idx

def render_dims(base_video):
    """(render_w, render_h, geom) capped at MAX_TW so cog/ltx/hy stay native and kling shrinks."""
    w, h = dims(base_video)
    tw = min(w, MAX_TW)
    th = int(round(h * tw / float(w)))
    return tw, th, geom_for(tw)

def load_prompt(key, bdir, dyn, pj):
    try:
        return pj[key]["prompts"][dyn].strip()
    except Exception:
        p = os.path.join(PDIR, bdir, dyn + ".txt")
        return open(p).read().strip() if os.path.exists(p) else ""


# ---------------- main ----------------
if os.path.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)
pj = json.load(open(os.path.join(PDIR, "prompts.json")))

sections = []   # (key, friendly, [(label, prompt, jpg)]) -- order here = order on the page

# ---- Kling v3 Pro: Base / VideoReward / Image (HPSv3), vr=true ----
kl_dyn = vr_dynamic(KLING_RES)
kl_prompts = json.load(open(KLING_PROMPTS))
kl_keep = sorted(n for n, v in kl_dyn.items() if v)
kl_items = []
for name in kl_keep:
    base = os.path.join(KLING_VID, KLING_BASE, name + ".mp4")
    vr   = os.path.join(KLING_VID, KLING_VR, name + ".mp4")
    img  = os.path.join(KLING_VID, KLING_IMAGE, name + ".mp4")
    if not (os.path.exists(base) and os.path.exists(vr) and os.path.exists(img)):
        print("  kling skip (missing variant):", name); continue
    tw, th, g = render_dims(base)
    rows = [("Base", base), ("VideoReward", vr), ("Image (HPSv3)", img)]
    jpg = "kling__%s.jpg" % name
    gw, gh, hi = build_strip(rows, tw, th, g, os.path.join(OUTDIR, jpg), "kling_" + name)
    kl_items.append((name.replace("_", " "), kl_prompts.get(name + ".mp4", ""), jpg))
    print("  kling %-32s %dx%d frames=%s" % (name, gw, gh, hi))
sections.append(("kling", "Kling v3 Pro", kl_items))
print("== Kling v3 Pro: %d / %d clips kept (vr dynamic) ==" % (len(kl_items), len(kl_dyn)))

# ---- CogVideoX: all check_v5 seeds, Base / VideoReward, vr=true ----
cog = json.load(open(COG_JSON))
cog_items = []
for S in sorted(cog, key=int):
    dynm = vr_dynamic(COG_RES % S)
    sdir = os.path.join(COG_VID_ROOT, "seed_%s" % S)
    pm = {"dyn_%03d" % f["prompt"]: f["video_propmt"].strip() for f in cog[S]["flips"]}
    for f in cog[S]["flips"]:
        dyn = "dyn_%03d" % f["prompt"]
        base = os.path.join(sdir, dyn + "_base.mp4")
        vr   = os.path.join(sdir, dyn + "_vr.mp4")
        if not dynm.get(dyn, False) or not os.path.exists(base):
            continue
        tw, th, g = render_dims(base)
        jpg = "cog__s%s_%s.jpg" % (S, dyn)
        gw, gh, hi = build_strip([("Base", base), ("VideoReward", vr)], tw, th, g,
                                 os.path.join(OUTDIR, jpg), "cog_%s_%s" % (S, dyn))
        cog_items.append(("seed %s &middot; %s" % (S, dyn), pm.get(dyn, ""), jpg))
        print("  cog seed %-4s %s %dx%d frames=%s" % (S, dyn, gw, gh, hi))
sections.append(("cog", "CogVideoX", cog_items))
print("== CogVideoX: %d clips across %d seeds ==" % (len(cog_items), len(cog)))

# ---- LTX-Video, HunyuanVideo: single dir each, Base / VideoReward, vr=true ----
ltx_dyn = vr_dynamic(LTX_RES)
hy_dyn = {}
for pat in HY_RES:
    hy_dyn.update(vr_dynamic(pat))
for key, bdir, friendly, dynm, seed in [
        ("ltx", "ltx_seed3", "LTX-Video", ltx_dyn, LTX_SEED),
        ("hy",  "hy_seed42", "HunyuanVideo", hy_dyn, HY_SEED)]:
    items = []
    for b in sorted(glob.glob(os.path.join(ROOT, bdir, "dyn_*_base.mp4"))):
        dyn = os.path.basename(b)[:-len("_base.mp4")]
        if not dynm.get("%s-%s" % (dyn, seed), False):
            continue
        base = os.path.join(ROOT, bdir, dyn + "_base.mp4")
        vr   = os.path.join(ROOT, bdir, dyn + "_vr.mp4")
        tw, th, g = render_dims(base)
        jpg = "%s__%s.jpg" % (key, dyn)
        gw, gh, hi = build_strip([("Base", base), ("VideoReward", vr)], tw, th, g,
                                 os.path.join(OUTDIR, jpg), "%s_%s" % (key, dyn))
        items.append((dyn, load_prompt(key, bdir, dyn, pj), jpg))
        print("  %-12s %s %dx%d frames=%s" % (bdir, dyn, gw, gh, hi))
    sections.append((key, friendly, items))
    print("== %s (%s): %d clips kept (vr dynamic) ==" % (friendly, bdir, len(items)))

# ---------------- index.html ----------------
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Base vs VideoReward &mdash; frame strips</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 24px auto; padding: 0 16px 60px; line-height: 1.5; }
  h1 { font-size: 23px; margin: 0 0 6px; }
  h2 { font-size: 18px; margin: 30px 0 2px; padding-top: 14px; border-top: 2px solid #bbb; }
  .sub { color: #888; font-size: 13px; margin: 0 0 6px; }
  .note { color: #777; font-size: 13px; margin-bottom: 8px; }
  .toc { font-size: 14px; margin: 8px 0 4px; }
  .toc a { margin-right: 14px; }
  .item { border-top: 1px solid #ddd; padding: 16px 0; }
  .row-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 13px; color: #444; font-weight: 600; }
  .prompt { margin: 5px 0 11px; color: #333; }
  img.strip { width: 100%; height: auto; border: 1px solid #eee; display: block; border-radius: 4px; }
  @media (prefers-color-scheme: dark) {
    body { background: #111; color: #eee; }
    .sub, .note, .row-id, .prompt { color: #bbb; }
    h2 { border-top-color: #444; }
    .item { border-top-color: #333; }
    img.strip { border-color: #333; }
  }
</style>
</head>
<body>
<h1>Base vs VideoReward &mdash; frame strips</h1>
<div class="note">Each composite stacks the same prompt rendered by <b>Base</b> (top) and <b>VideoReward</b>;
the <b>Kling v3 Pro</b> section adds a third row, <b>Image (HPSv3)</b> (the image-reward variant). Only clips
whose <b>VideoReward</b> render is actually dynamic (VBench <code>dynamic_degree = 1</code>) are shown. Four
frames are sampled at equal time points (start / &frac13; / &frac23; / end): for the 121-frame clips
(LTX-Video, HunyuanVideo, Kling v3 Pro) these are frames <b>0 / 40 / 80 / 120</b>; the CogVideoX clips have
81 frames, so the same points map to frames 0 / 27 / 53 / 80, shown across several downstream seeds (the seed
is labeled on each row). The exact frame index is printed above each column. Frames are shown at native
resolution (capped at 480&thinsp;px wide).</div>
__TOC__
"""

SEC = """<h2 id="__KEY__">__FRIENDLY__ <span class="sub">&middot; __N__ clips</span></h2>
"""
ITEM = """<section class="item">
  <div class="row-id">__RID__</div>
  <div class="prompt">__PROMPT__</div>
  <img class="strip" src="frames/__IMG__" alt="__RID__ frame strip" loading="lazy">
</section>
"""

toc = '<div class="toc"><b>Backbones:</b> ' + " ".join(
    '<a href="#{}">{}</a>'.format(k, f) for k, f, _ in sections) + "</div>"
body = HEAD.replace("__TOC__", toc)
for key, friendly, items in sections:
    body += SEC.replace("__KEY__", key).replace("__FRIENDLY__", html.escape(friendly)).replace("__N__", str(len(items)))
    for label, prompt, img in items:
        rid = "{} &middot; {}".format(friendly, label)
        body += (ITEM.replace("__RID__", rid)
                     .replace("__PROMPT__", html.escape(prompt))
                     .replace("__IMG__", img))
body += "\n</body>\n</html>\n"

with open(os.path.join(WORK, "index.html"), "w") as fh:
    fh.write(body)
total = sum(len(it) for _, _, it in sections)
print("index.html written:", total, "strips ::", ", ".join("%s=%d" % (f, len(it)) for _, f, it in sections))
