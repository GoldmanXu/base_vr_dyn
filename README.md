# base_vr_dyn — Base vs VideoReward frame strips

Static frame-strip comparison of **Base** vs **VideoReward** across three text-to-video backbones
(**CogVideoX**, **LTX-Video**, **HunyuanVideo**) on dynamic-motion prompts.

**Live page:** https://goldmanxu.github.io/base_vr_dyn/

## Layout

Each composite (one per prompt) has:

- **top row = Base**, **bottom row = VideoReward**
- four frames sampled at equal time points (start / ⅓ / ⅔ / end)
  - 24-fps clips (LTX-Video, HunyuanVideo, 121 frames) → frames **0 / 40 / 80 / 120**
  - CogVideoX clips (81 frames) → the same time points → frames **0 / 27 / 53 / 80**
- the exact frame index is printed above each column

## Contents

| path | what |
|---|---|
| `index.html` | the demo page (grouped by backbone) |
| `frames/<key>__dyn_NNN.jpg` | baked composite strips, native resolution, JPEG q92 (`key` = `cog` / `ltx` / `hy`) |
| `scripts/build_base_vr_frames.py` | generator (ffmpeg frame extraction + Pillow compositing) |

82 strips total: CogVideoX 58 (check_v5; seeds 1/2/3/777/1024/4096), LTX-Video 18 (seed 3), HunyuanVideo 6 (seed 42).

## Regenerate

Run on the box that holds the source videos (`/data_2/xujinyuan/vbench_check/base0_vr1_videos/`):

```bash
/data/xujingyuan/envs/verl/bin/python scripts/build_base_vr_frames.py
```

This rewrites `frames/` and `index.html`. Frames are pasted at native resolution and saved as JPEG q92,
so the strips stay sharp. The repo is ~24 MB (well under the GitHub Pages 1 GB limit), so everything lives
on `main` and is served directly by Pages — no jsDelivr / assets branch needed.
