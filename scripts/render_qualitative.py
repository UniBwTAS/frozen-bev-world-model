# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
# SPDX-License-Identifier: MIT
"""render_qualitative.py — single-source renderer for every qualitative
forecast asset, so the paper figure and the animated GIFs share identical
color semantics by construction.

One STYLE block defines the look (matte segmentation palette, backdrop,
map opacity, detection/reference line colors, legend rows). Two artifacts
can be produced from the same NPZ inputs:

  --artifacts figure   static strip for the paper (camera + forecast
                       rollouts at +0.5/+1.5/+3.0 s + a measured-future
                       reference panel + the two-row legend)
  --artifacts frames   one PNG frame per forecast step for the web duet
                       animation (camera-rig layout comes from
                       render_forecast_hero_combo.py; this script emits the
                       same panels/colors in duet layout) and, with --gif,
                       assembles them via ffmpeg (GIF + H.265/yuv444p
                       full-range MP4)

Inputs are the world-model NPZ chain used everywhere else:
seg rollout NPZ + decoded detection NPZs (pred/real) + the nuScenes data
root; --tokens fixes the forecast window.

Documented in the companion README (Rendering section).
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

# ----------------------------------------------------------------------------
# STYLE — the single source of truth for all qualitative forecast renders
# ----------------------------------------------------------------------------
MATTE_PALETTE = ["#D39D3A", "#7BA7C7", "#6BA78B", "#E8D27D", "#407095", "#BA561B"]
MAP_CLASSES = ["drivable area", "pedestrian crossing", "walkway",
               "stop line", "carpark area", "divider"]
BACKDROP = "#FEFCFE"          # no-data backdrop (matches paper + web)
MAP_ALPHA = 0.84              # map-over-backdrop opacity (matte, no haze)
UPSCALE = 6                   # sub-pixel raster supersampling (crisp edges)
EGO_COLOR = "#222222"
DET_COLOR = "#FF3D3D"         # forecast detections (red)
ORACLE_COLOR = "#00B7EB"      # measured detections (cyan)
GT_COLOR = "#444444"          # ground-truth boxes (dashed gray)
LEGEND_DETECTIONS = [
    (EGO_COLOR, "ego vehicle", False),
    (DET_COLOR, "forecast (decoded)", False),
    (ORACLE_COLOR, "measured (decoded)", False),
    (GT_COLOR, "ground truth", True),
]
CREAM = None                  # resolved from BACKDROP at render time
XHALF = 30.5                  # portrait BEV crop for the duet frames
SCORE = 0.2                   # detection score threshold


def _init_colors():
    global CREAM
    import matplotlib.colors as mc
    CREAM = np.array(mc.to_rgb(BACKDROP))


# ----------------------------------------------------------------------------
# shared drawing (identical for both artifacts)
# ----------------------------------------------------------------------------
def upscale(mask, f=UPSCALE):
    return np.repeat(np.repeat(mask, f, axis=1), f, axis=2)


def map_layer(mask):
    """Baked (opaque) matte map layer: alpha MAP_ALPHA over BACKDROP."""
    import matplotlib.colors as mc
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    import render_forecast_hero_combo as R
    rgba = R.mask_img(upscale(mask), MAP_ALPHA)
    a = rgba[..., 3:4]
    return np.clip(rgba[..., :3] * a + (1 - a) * CREAM, 0, 1)


def draw_map(ax, mask, extent, red=None, cyan=None, gt=None, lw=1.6):
    import matplotlib.pyplot as plt
    x0, x1 = extent
    ax.imshow(map_layer(mask), extent=(x0, x1, x1, x0),
              interpolation="nearest", origin="upper", zorder=1)
    ax.set_xlim(x0, x1); ax.set_ylim(x1, x0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(EGO_COLOR); sp.set_linewidth(1.2)
    if red is not None:
        R.draw_boxes_bev(ax, red[0], DET_COLOR, lw,
                         score_thresh=red[2], scores=red[1])
    if cyan is not None:
        R.draw_boxes_bev(ax, cyan[0], ORACLE_COLOR, lw - 0.1,
                         score_thresh=cyan[2], scores=cyan[1])
    if gt is not None:
        R.draw_boxes_bev(ax, gt, GT_COLOR, lw - 0.5, dashed=True)
    ax.add_patch(plt.Rectangle((-1.1, -2.2), 2.2, 4.4, fill=False,
                               ec=EGO_COLOR, lw=0.9, zorder=6))


def legend_band(fig, y0, height, compact=False):
    """Two-row open legend: detections (lines) + map classes (blocks)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    f1 = 7.1 if not compact else 6.2
    f2 = f1
    _hd = [Line2D([], [], color=c, lw=1.9, label=lab,
                  ls="--" if d else "-") for c, lab, d in LEGEND_DETECTIONS]
    _hm = [Patch(fc=np.clip(np.array(matplotlib.colors.to_rgb(col))
                            * MAP_ALPHA + (1 - MAP_ALPHA) * CREAM, 0, 1),
                 ec="#b9b9b9" if lab == "no data" else "none", label=lab)
           for col, lab in zip(MATTE_PALETTE, MAP_CLASSES + ["no data"])]
    f1 = f2 = f1
    leg1 = fig.legend(handles=_hd, loc="lower center", ncol=4, frameon=False,
                      fontsize=f1, handlelength=1.5, columnspacing=1.15,
                      bbox_to_anchor=(0.5, y0 + height * 0.55))
    leg2 = fig.legend(handles=_hm, loc="lower center", ncol=7, frameon=False,
                      fontsize=f2, handlelength=0.85, columnspacing=0.8,
                      bbox_to_anchor=(0.5, y0 + height * 0.12))
    return leg1, leg2


# ----------------------------------------------------------------------------
# artifact: paper figure (static strip)
# ----------------------------------------------------------------------------
def render_figure(sc, steps, agg_json, out_png):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    fig = plt.figure(figsize=(7.16, 3.05), dpi=300)
    nforecast = len(steps)
    gs = fig.add_gridspec(1, nforecast + 2,
                          width_ratios=[0.98] + [1.01] * (nforecast + 1),
                          wspace=0.07, left=0.004, right=0.996,
                          top=0.875, bottom=0.16)
    E = (-50.0, 50.0)
    img = mpimg.imread(sc["cam"])
    Hc, Ww = img.shape[:2]
    ch = int(Hc * 0.70); r0 = int(Hc * 0.18)
    cw = int(ch * 1.35); c0 = (Ww - cw) // 2
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img[r0:r0 + ch, c0:c0 + cw]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(EGO_COLOR); sp.set_linewidth(1.1)
    ax.set_title("(a) Camera context", fontsize=9.0, pad=3)

    for j, k in enumerate(steps):
        step = 2 + k
        axk = fig.add_subplot(gs[0, 1 + j])
        red = (sc["pr"]["boxes_%d" % step], sc["pr"]["scores_%d" % step], SCORE)
        draw_map(axk, sc["seg"]["dynamic_pred"][k - 1], (E[0], E[1]), red=red)
        found, nn = R.match_gt(sc["pr"]["boxes_%d" % step],
                               sc["pr"]["scores_%d" % step],
                               sc["pr"]["labels_%d" % step],
                               sc["tb_%d" % k]["gt_boxes"],
                               sc["tb_%d" % k]["gt_names"],
                               dist=2.0, score_thresh=SCORE)
        axk.set_title("(%s) Forecast at $+%.1f$ s (decoded)"
                      % ("bcd"[j], 0.5 * k), fontsize=9.0, pad=3,
                      color=DET_COLOR)
        axk.text(0.04, 0.04, "mIoU %.2f\n%d/%d"
                 % (float(sc["seg"]["dynamic_mious"][k - 1]), found, nn),
                 transform=axk.transAxes, fontsize=7.4, color=DET_COLOR,
                 va="bottom", zorder=8, linespacing=1.18,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white",
                           ec="#b9b9b9", lw=0.8, alpha=0.94))

    kk = steps[-1]
    step = 2 + kk
    axm = fig.add_subplot(gs[0, nforecast + 1])
    draw_map(axm, sc["seg"]["dynamic_real"][kk - 1], (E[0], E[1]),
             cyan=(sc["rl"]["boxes_%d" % step], sc["rl"]["scores_%d" % step],
                   SCORE),
             gt=sc["tb_%d" % kk]["gt_boxes"])
    axm.set_title("(%s) Measured at $+%.1f$ s" % ("efgh"[nforecast - 1],
                  0.5 * kk), fontsize=9.0, pad=3, color=ORACLE_COLOR)
    legend_band(fig, 0.0, 0.15)
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    print("[figure] ->", out_png)


def render_frames(sc, steps, agg_json, frames_dir, k_total):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    frames_dir.mkdir(exist_ok=True)
    E = (-50.0, 50.0)
    XHALF = 30.5
    img = mpimg.imread(sc["cam"])
    Hc, Ww = img.shape[:2]
    ch = int(Hc * 0.70); r0 = int(Hc * 0.18); cw = int(ch * 1.35); c0 = (Ww - cw) // 2
    for fi, k in enumerate(steps):
        step = 2 + k
        fig = plt.figure(figsize=(7.16, 3.14), dpi=300)
        gs = fig.add_gridspec(1, 3, width_ratios=[0.98, 1.01, 1.01],
                              wspace=0.05, left=0.004, right=0.996,
                              top=0.90, bottom=0.115)
        axc = fig.add_subplot(gs[0, 0])
        axc.imshow(img[r0:r0 + ch, c0:c0 + cw])
        axc.set_xticks([]); axc.set_yticks([])
        for sp in axc.spines.values(): sp.set_color(EGO_COLOR); sp.set_linewidth(1.2)
        axc.set_title("(a) Camera context", fontsize=9.4, pad=3)
        draw_map(fig.add_subplot(gs[0, 1]),
                 sc["seg"]["dynamic_pred"][k - 1], (-XHALF, XHALF, YLIM, -YLIM),
                 red=(sc["pr"]["boxes_%d" % step], sc["pr"]["scores_%d" % step], SCORE))
        found, nn = R.match_gt(sc["pr"]["boxes_%d" % step],
                               sc["pr"]["scores_%d" % step],
                               sc["pr"]["labels_%d" % step],
                               sc["tb_%d" % k]["gt_boxes"],
                               sc["tb_%d" % k]["gt_names"],
                               dist=2.0, score_thresh=SCORE)
        axb = fig.axes[-1]
        axb.set_title("(b) Forecast at $+%.1f$ s (decoded)" % (0.5 * k),
                      fontsize=9.4, pad=3, color=DET_COLOR)
        axb.text(0.045, 0.045, "forecast (world model)\nmIoU %.2f\nmatched %d/%d ($\\leq$ 2 m)"
                 % (float(sc["seg"]["dynamic_mious"][k - 1]), found, nn),
                 transform=axb.transAxes, fontsize=7.8, color=DET_COLOR,
                 va="bottom", zorder=8, linespacing=1.18,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white",
                           ec="#b9b9b9", lw=0.8, alpha=0.94))
        axm = fig.add_subplot(gs[0, 2])
        axm = fig.add_subplot(gs[0, 2])
        draw_map(axm, sc["seg"]["dynamic_real"][k - 1],
                 (-XHALF, XHALF, YLIM, -YLIM),
                 cyan=(sc["rl"]["boxes_%d" % step], sc["rl"]["scores_%d" % step], SCORE),
                 gt=sc["tb_%d" % k]["gt_boxes"])
        axm.set_title("(c) Measured at $+%.1f$ s (decoded)" % (0.5 * k),
                      fontsize=9.4, pad=3, color=ORACLE_COLOR)
        axm.text(0.045, 0.045, "real latent (reference)\ndecode of the\nmeasured future",
                 transform=axm.transAxes, fontsize=7.8, color=ORACLE_COLOR,
                 va="bottom", zorder=8, linespacing=1.18,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white",
                           ec="#b9b9b9", lw=0.8, alpha=0.94))
        legend_band(fig, 0.0, 0.115, compact=True)
        fig.savefig(frames_dir / ("frame%03d.png" % fi), dpi=300,
                    facecolor="white")
        plt.close(fig)
    print("[frames] %d frames -> %s" % (len(steps), frames_dir))


def assemble_gif(frames_dir, out_gif, width=1500):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "2",
                    "-i", str(frames_dir / "frame%03d.png"),
                    "-vf", "scale=%d:-1:flags=lanczos,palettegen" % width,
                    str(Path(out_gif).with_suffix(".palette.png"))], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "2",
                    "-i", str(frames_dir / "frame%03d.png"),
                    "-i", str(Path(out_gif).with_suffix(".palette.png")),
                    "-lavfi", "scale=%d:-1:flags=lanczos [x]; [x][1:v] paletteuse=bayer_scale=4" % width,
                    str(out_gif)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "2",
                    "-i", str(frames_dir / "frame%03d.png"),
                    "-vf", "scale=out_range=full",
                    "-c:v", "libx265", "-crf", "14", "-pix_fmt", "yuv444p",
                    "-color_range", "2", "-tag:v", "hvc1",
                    "-movflags", "+faststart",
                    str(Path(out_gif).with_suffix(".h265_full.mp4"))], check=True)


def _resolve_cam(data_root, rel):
    """BEVFusion PKL paths are relative ("./data/nuscenes/...")."""
    if rel.startswith("./data/nuscenes/"):
        rel = rel[len("./data/nuscenes/"):]
    import matplotlib  # noqa: F401
    p = Path(data_root) / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="figure",
                    help="figure | frames | all")
    ap.add_argument("--seg-npz", required=True)
    ap.add_argument("--pred-npz", required=True)
    ap.add_argument("--real-npz", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokens", required=True,
                    help="token list, one per line (context + forecast)")
    ap.add_argument("--agg-json", default=None)
    ap.add_argument("--figure-out", default="figs/qualitative_forecast.png")
    ap.add_argument("--frames-out", default="build/qualitative_frames")
    ap.add_argument("--gif-out", default=None)
    ap.add_argument("--steps", default="1,3,6",
                    help="comma list of forecast steps shown (paper)")
    args = ap.parse_args()
    _init_colors()

    global R
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import render_forecast_hero_combo as R
    R.OI[:] = MATTE_PALETTE

    sc = {"seg": np.load(args.seg_npz, allow_pickle=True),
          "pr": np.load(args.pred_npz, allow_pickle=True),
          "rl": np.load(args.real_npz, allow_pickle=True),
          "cam": _resolve_cam(args.data_root, str(np.load(args.seg_npz, allow_pickle=True)["dynamic_cams"][2])),
          }
    sc["tokens"] = [str(t) for t in sc["seg"]["dynamic_tokens"]]
    infos = pickle.load(open(Path(args.data_root) / "nuscenes_infos_val.pkl",
                             "rb"))["infos"]
    by = {i["token"]: i for i in infos}
    for k in range(1, int(sc["seg"]["k"]) + 1):
        sc["tb_%d" % k] = by[sc["tokens"][2 + k]]
    sc["agg"] = args.agg_json
    sc["k_total"] = 21

    arts = {x.strip() for x in args.artifacts.split(",")}
    steps = [int(x) for x in args.steps.split(",")]
    if "figure" in arts:
        render_figure(sc, steps, args.agg_json, args.figure_out)
        print("[figure] ->", args.figure_out)
    if "frames" in arts:
        frames_dir = Path(args.frames_out)
        render_frames(sc, list(range(1, 9)), args.agg_json, frames_dir, 21)
        if args.gif_out:
            assemble_gif(frames_dir, args.gif_out)
            print("[gif] ->", args.gif_out)


if __name__ == "__main__":
    main()
