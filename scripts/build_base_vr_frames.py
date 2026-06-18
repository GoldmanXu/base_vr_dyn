#!/usr/bin/env python
# Build base-vs-VideoReward frame-strip demo (time-normalized: start / 1/3 / 2/3 / end).
# One composite per (backbone, clip): top row Base, bottom row VideoReward, 4 frames.
# Frames are pasted at NATIVE resolution (no downscale) and saved as JPEG q92, so the strips stay sharp.
import json, os, glob, html, shutil, subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT   = "/data_2/xujinyuan/vbench_check/base0_vr1_videos"
PDIR   = os.path.join(ROOT, "propmt")
WORK   = "/data_2/xujinyuan/vbench_check/base_vr_dyn_repo"
OUTDIR = os.path.join(WORK, "frames")

# cog = the check_v5 CogVideoX run, ALL seeds (58 pairs across 6 seeds, 448x768, 81 frames).
# Videos live in per-seed dirs outside ROOT; prompts come from a combined by-seed eval JSON.
COG_VID_ROOT = "/data_2/xujinyuan/vbench_check/eval_check_v5_cogvideox/base0_vr1_videos"   # seed_<S>/
COG_JSON     = "/data_2/xujinyuan/vbench_check/eval_check_v5_cogvideox/base0_vr1_by_seed.json"

# (key, dir, friendly name) for the single-dir backbones. cog is handled separately (multi-seed).
BACKBONES = [
    ("ltx", "ltx_seed3", "LTX-Video"),
    ("hy",  "hy_seed42", "HunyuanVideo"),
]
VARIANTS = [("base", "Base"), ("vr", "VideoReward")]   # suffix, label
FRACS = [0.0, 1.0/3.0, 2.0/3.0, 1.0]                   # equal time points
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
JPEG_Q = 92

# reference geometry defined at a 200px frame width; scaled to native width so
# proportions stay identical while pixels stay native-sharp.
TW_REF = 200
G0 = dict(GAPF=4, LBL=150, LBLGAP=8, HDR=30, GAPR=8, EDGE=10, FH=18, FL=19)

def nframes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nw=1:nk=1", path], capture_output=True, text=True).stdout.strip()
    return int(out)

def dims(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)

def idxs(n):
    return [int(round(f * (n - 1))) for f in FRACS]

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

def load_prompt(key, bdir, dyn, pj):
    try:
        return pj[key]["prompts"][dyn].strip()
    except Exception:
        p = os.path.join(PDIR, bdir, dyn + ".txt")
        return open(p).read().strip() if os.path.exists(p) else ""

def geom_for(w):
    s = w / float(TW_REF)
    return {k: max(1, int(round(v * s))) for k, v in G0.items()}

def build_strip(bdir, dyn, w, h, g, out_jpg):
    TW, TH = w, h
    f_hdr = ImageFont.truetype(FONT_B, g["FH"])
    f_lbl = ImageFont.truetype(FONT_B, g["FL"])
    frames_w = 4 * TW + 3 * g["GAPF"]
    GW = g["EDGE"] + g["LBL"] + g["LBLGAP"] + frames_w + g["EDGE"]
    GH = g["EDGE"] + g["HDR"] + 2 * TH + g["GAPR"] + g["EDGE"]
    canvas = Image.new("RGB", (GW, GH), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    frames_x0 = g["EDGE"] + g["LBL"] + g["LBLGAP"]
    header_idx = None
    for r, (suf, vlabel) in enumerate(VARIANTS):
        vid = os.path.join(ROOT, bdir, "{}_{}.mp4".format(dyn, suf))
        ind = idxs(nframes(vid))
        if header_idx is None:
            header_idx = ind
            for c, fi in enumerate(ind):
                cx = frames_x0 + c * (TW + g["GAPF"]) + TW / 2
                ctext(draw, cx, g["EDGE"] + g["HDR"] / 2, "frame {}".format(fi), f_hdr)
        tmp = "/tmp/bvf_{}_{}_{}".format(bdir.strip("/").replace("/", "_"), dyn, suf)
        pngs = extract(vid, ind, tmp)
        y = g["EDGE"] + g["HDR"] + r * (TH + g["GAPR"])
        ctext(draw, g["EDGE"] + g["LBL"] / 2, y + TH / 2, vlabel, f_lbl)
        for c, p in enumerate(pngs):
            im = Image.open(p).convert("RGB")
            if im.size != (TW, TH):                       # paste native; resize only if mismatch
                im = im.resize((TW, TH), Image.LANCZOS)
            canvas.paste(im, (int(frames_x0 + c * (TW + g["GAPF"])), int(y)))
        shutil.rmtree(tmp)
    canvas.save(out_jpg, quality=JPEG_Q, optimize=True, progressive=True)
    return GW, GH, header_idx

# ---- main ----
if os.path.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)
pj = json.load(open(os.path.join(ROOT, "propmt", "prompts.json")))

sections = []   # (key, friendly, [ (label, prompt, jpg_name) ])

# ---- CogVideoX: all check_v5 seeds (label carries the seed) ----
cogjd = json.load(open(COG_JSON))
cog_items = []
for S in sorted(cogjd.keys(), key=int):                 # 1, 2, 3, 777, 1024, 4096
    sdir = os.path.join(COG_VID_ROOT, "seed_{}".format(S))
    bases = sorted(glob.glob(os.path.join(sdir, "dyn_*_base.mp4")))
    dyns = [os.path.basename(b)[:-len("_base.mp4")] for b in bases]
    pm = {"dyn_%03d" % f["prompt"]: f["video_propmt"].strip() for f in cogjd[S]["flips"]}
    for dyn in dyns:
        w, h = dims(os.path.join(sdir, dyn + "_base.mp4"))
        g = geom_for(w)
        jpg_name = "cog__s{}_{}.jpg".format(S, dyn)
        gw, gh, hi = build_strip(sdir, dyn, w, h, g, os.path.join(OUTDIR, jpg_name))
        cog_items.append(("seed {} &middot; {}".format(S, dyn), pm.get(dyn, ""), jpg_name))
        print("  cog seed {:<4} {} {}x{} frames={}".format(S, dyn, gw, gh, hi))
sections.append(("cog", "CogVideoX", cog_items))
print("== CogVideoX: {} clips across {} seeds ==".format(len(cog_items), len(cogjd)))

# ---- LTX-Video, HunyuanVideo: single dir each ----
for key, bdir, friendly in BACKBONES:
    bases = sorted(glob.glob(os.path.join(ROOT, bdir, "dyn_*_base.mp4")))
    dyns = [os.path.basename(b)[:-len("_base.mp4")] for b in bases]
    w, h = dims(os.path.join(ROOT, bdir, dyns[0] + "_base.mp4"))
    g = geom_for(w)
    items = []
    for dyn in dyns:
        jpg_name = "{}__{}.jpg".format(key, dyn)
        gw, gh, hi = build_strip(bdir, dyn, w, h, g, os.path.join(OUTDIR, jpg_name))
        items.append((dyn, load_prompt(key, bdir, dyn, pj), jpg_name))
        print("  {:<14} {} {}x{} frames={}".format(bdir, dyn, gw, gh, hi))
    sections.append((key, friendly, items))
    print("== {} ({}): {} prompts, native {}x{} ==".format(friendly, bdir, len(items), w, h))

# ---------- index.html ----------
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Base vs VideoReward — frame strips</title>
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
<div class="note">Each composite: <b>top row = Base</b>, <b>bottom row = VideoReward</b>. Four frames are sampled at
equal time points (start / &frac13; / &frac23; / end). For the 24-fps clips (LTX-Video, HunyuanVideo) these are frames
<b>0 / 40 / 80 / 120</b>; the CogVideoX clips have 81 frames, so the same time points map to frames 0 / 27 / 53 / 80,
and CogVideoX is shown across several downstream seeds (the seed is labeled on each row).
The exact frame index is printed above each column. Frames are shown at native resolution.</div>
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
print("index.html written:", total, "strips")
