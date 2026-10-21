# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
"""render_qualitative.py — single source for the qualitative forecast assets.

Two artifacts from one STYLE block, so the paper figure and the animated
web GIFs share identical map colors by construction:

  --artifacts figure   approved five-panel strip (camera | forecast steps |
                       measured reference) — layout copied 1:1 from the
                       render that produced the committed paper figure
                       (measured reference: figs/qualitative_forecast.png);
                       fonts, margins, badges and legend rows unchanged.
  --artifacts frames   one PNG per step in the six-camera-rig duet layout —
                       geometry copied 1:1 from render_forecast_hero_combo.py
                       (rig rect grid, corner sparkline, honest badges); the
                       only change is the map layer: the shared matte map
                       below replaces combo's raw color overlay.
  --gif (with frames)  assembles GIF + full-range H.265 MP4 directly from the
                       PNG frames (two-pass palette). No intermediate lossy
                       MP4 anything — the old chain encoded yuv420 first and
                       quantized from that, which is where the color drift
                       against the paper figure came from.

Inputs: seg rollout NPZ + decoded detection NPZs (pred/real) + nuScenes root.
    python3 scripts/render_qualitative.py \
        --artifacts all \
        --seg-npz seg_<scene>.npz \
        --pred-npz dets_<scene>_pred.npz --real-npz dets_<scene>_real.npz \
        --data-root path/to/nuscenes/complete \
        [--agg-json agg_decay.json] [--steps 1,3,6] [--k-shown 8]
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_forecast_hero_combo as R   # noqa: E402  (shared drawing helpers)

# ---------------------------------------------------------------------------
# STYLE — single source of truth for every qualitative forecast render.
# Matte segmentation palette baked over the warm cream backdrop (#FEFCFE):
#   drivable #D39D3A · ped. crossing #7BA7C7 · walkway #6BA78B
#   stop line #E8D27D · carpark #407095 · divider #BA561B
# The combo module's palette is overridden in place so its box/line colors
# and the GIF matte look stay byte-consistent everywhere.
# ---------------------------------------------------------------------------
R.OI[:] = ["#D39D3A", "#7BA7C7", "#6BA78B", "#E8D27D", "#407095", "#BA561B"]
CREAM = np.array([0.996, 0.988, 0.996])   # BEV backdrop #FEFCFE (user-provided)
GIF_SCALE = 0.42   # effective raster scale of the web GIFs (soft pastel cells)
SCORE = 0.2
XLIM = 30.5        # zoomed corridor (portrait 61 x 100 m slot)
YLIM = 50.0
EGO_COLOR = "#222222"
GT_COLOR = "#444444"
EDGE_COLOR = "#222222"


def upmask(m, f=6):
    """Nearest 6x supersampling: renders the 0.5 m cells as sub-pixel tiles."""
    return np.repeat(np.repeat(m, f, axis=1), f, axis=2)


# Legend content — ONE source for the paper strip and the rig GIF: edit the
# labels, colors or chip scaling here and both artifacts update together.
LEGEND_MAP_CLASSES = ["drivable area", "pedestrian crossing", "walkway",
                      "stop line", "carpark area", "divider"]


def legend_handles():
    """Two-row open legend: detections (lines) row + matte map-class chips."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hbox = [
        Line2D([], [], color=EGO_COLOR, lw=1.9, label="ego vehicle"),
        Line2D([], [], color=R.DET_COLOR, lw=1.9,
               label="forecast detections (decoded)"),
        Line2D([], [], color=R.ORACLE_COLOR, lw=1.9,
               label="measured detections (decoded)"),
        Line2D([], [], color=GT_COLOR, lw=1.5, ls="--",
               label="ground truth (dashed gray)"),
    ]
    hmap = [Patch(fc=np.clip(np.array(matplotlib.colors.to_rgb(col)) * 0.84
                             + 0.16 * CREAM, 0, 1), ec="none", label=lb)
            for col, lb in zip(R.OI, LEGEND_MAP_CLASSES)]
    hmap.append(Patch(fc=tuple(CREAM), ec="#b9b9b9", lw=0.5, label="no data"))
    return hbox, hmap


def draw_legend_band(fig, fontsize, y_lines, y_chips):
    """Place the shared legend band with per-artifact geometry."""
    hbox, hmap = legend_handles()
    fig.legend(handles=hbox, loc="lower center", ncol=4, frameon=False,
               fontsize=fontsize, handlelength=1.55, columnspacing=1.35,
               bbox_to_anchor=(0.5, y_lines))
    fig.legend(handles=hmap, loc="lower center", ncol=7, frameon=False,
               fontsize=fontsize, handlelength=0.85, columnspacing=0.9,
               bbox_to_anchor=(0.5, y_chips))


def matte_map(m):
    """GIF-grade raster: downscale the class overlay to the GIF's effective
    resolution, then bilinear upscaled display gives the soft pastel edges
    of the animated GIFs while boxes/text stay sharp."""
    up = upmask(m)
    nh = max(2, int(up.shape[1] * GIF_SCALE))
    nw = max(2, int(up.shape[2] * GIF_SCALE))
    from PIL import Image as _I
    lo = np.zeros((up.shape[0], nh, nw), dtype=bool)
    for c in range(up.shape[0]):
        lo[c] = np.array(_I.fromarray(up[c].astype(np.uint8) * 255)
                         .resize((nw, nh), _I.NEAREST)) > 127
    rgba = R.mask_img(lo, 0.80)
    a = rgba[..., 3:4]
    rgb = (rgba[..., :3] * a) + (1 - a) * CREAM
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# scene loader
# ---------------------------------------------------------------------------
def load_scene(a):
    seg = np.load(a.seg_npz, allow_pickle=True)
    pr = np.load(a.pred_npz, allow_pickle=True) if a.pred_npz else None
    rl = np.load(a.real_npz, allow_pickle=True) if a.real_npz else None
    infos = pickle.load(open(Path(a.data_root) / "nuscenes_infos_val.pkl",
                             "rb"))["infos"]
    agg = json.load(open(a.agg_json)) if a.agg_json else None
    return {"seg": seg, "pr": pr, "rl": rl, "infos": infos, "agg": agg,
            "tokens": [str(t) for t in seg["dynamic_tokens"]],
            "cam": str(seg["dynamic_cams"][2])}


# ---------------------------------------------------------------------------
# artifact: figure — exact layout of the approved five-panel strip
# (camera | +0.5 s | +1.5 s | +3.0 s forecasts | measured reference at +3 s)
# ---------------------------------------------------------------------------
def render_figure(sc, steps, out_png):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    by = {i["token"]: i for i in sc["infos"]}
    NCOL = 1 + len(steps) + 1
    fig = plt.figure(figsize=(7.16, 3.14), dpi=300)
    gs = fig.add_gridspec(1, NCOL, wspace=0.06, left=0.004, right=0.996,
                          top=0.895, bottom=0.082)

    def bev(ax, m, red=None, cyan=None, gt=None, lw=1.25):
        ax.imshow(matte_map(m),
                  extent=(-YLIM, YLIM, YLIM, -YLIM),
                  interpolation="bilinear", origin="upper", zorder=1)
        if red is not None:
            R.draw_boxes_bev(ax, red[0], R.DET_COLOR, lw,
                             score_thresh=SCORE, scores=red[1])
        if cyan is not None:
            R.draw_boxes_bev(ax, cyan[0], R.ORACLE_COLOR, lw - 0.1,
                             score_thresh=SCORE, scores=cyan[1])
        if gt is not None:
            R.draw_boxes_bev(ax, gt, GT_COLOR, lw - 0.35, dashed=True)
        ax.add_patch(plt.Rectangle((-1.1, -2.2), 2.2, 4.4, fill=False,
                                   ec=EDGE_COLOR, lw=0.8, zorder=6))
        ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-YLIM, YLIM)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(EGO_COLOR); sp.set_linewidth(1.1)

    img = mpimg.imread(sc["cam"])
    H, W = img.shape[:2]
    # camera crop to a portrait ~0.70 aspect to match the BEV panels
    ar = (2 * XLIM) / (2 * YLIM)
    cw = min(W, int(H * ar)); c0 = (W - cw) // 2
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img[:, c0:c0 + cw]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(EGO_COLOR); sp.set_linewidth(1.1)
    ax.set_title("(a) Camera context", fontsize=9.0, pad=3)
    ax.text(0.04, 0.968, "Busy intersection", transform=ax.transAxes,
            fontsize=7.8, color="#111111", va="top", zorder=8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#b9b9b9",
                      lw=0.8, alpha=0.94))

    for c, k in enumerate(steps, start=1):
        idx = 2 + k
        tb = by[sc["tokens"][idx]]
        found, n_ = R.match_gt(sc["pr"]["boxes_%d" % idx],
                               sc["pr"]["scores_%d" % idx],
                               sc["pr"]["labels_%d" % idx],
                               tb["gt_boxes"], tb["gt_names"],
                               dist=2.0, score_thresh=SCORE)
        axk = fig.add_subplot(gs[0, c])
        bev(axk, sc["seg"]["dynamic_pred"][k - 1],
            red=(sc["pr"]["boxes_%d" % idx], sc["pr"]["scores_%d" % idx]))
        axk.set_title("(%s) Forecast $+%0.1f$ s"
                      % (chr(98 + c - 1), 0.5 * k), fontsize=9.0, pad=3,
                      color=R.DET_COLOR)
        axk.text(0.04, 0.04, "mIoU %.2f\n%d/%d"
                 % (float(sc["seg"]["dynamic_mious"][k - 1]), found, n_),
                 transform=axk.transAxes, fontsize=7.6, color=R.DET_COLOR,
                 va="bottom", zorder=8, linespacing=1.18,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white",
                           ec="#b9b9b9", lw=0.8, alpha=0.94))

    idx = 2 + steps[-1]
    axr = fig.add_subplot(gs[0, NCOL - 1])
    bev(axr, sc["seg"]["dynamic_real"][steps[-1] - 1],
        cyan=(sc["rl"]["boxes_%d" % idx], sc["rl"]["scores_%d" % idx]),
        gt=by[sc["tokens"][idx]]["gt_boxes"])
    axr.set_title("(%s) Measured, $+%.1f$ s"
                  % (chr(98 + len(steps)), 0.5 * steps[-1]),
                  fontsize=9.0, pad=3, color=R.ORACLE_COLOR)
    axr.text(0.04, 0.04, "decode of the\nmeasured\nfuture latent",
             transform=axr.transAxes, fontsize=7.6, color=R.ORACLE_COLOR,
             va="bottom", zorder=8, linespacing=1.18,
             bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#b9b9b9",
                       lw=0.8, alpha=0.94))

    draw_legend_band(fig, 7.1, 0.058, 0.012)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    print("[figure] ->", out_png)


# ---------------------------------------------------------------------------
# artifact: frames — six-camera rig, geometry copied 1:1 from
# render_forecast_hero_combo.py; the map layer is the shared matte map.
# ---------------------------------------------------------------------------
def render_frames(sc, fps, k_shown, frames_dir, no_dets=False):
    dg, pr, rr = sc["seg"], sc["pr"], sc["rl"]
    tokens, infos, agg = sc["tokens"], sc["infos"], sc["agg"]
    n_input, K = int(dg["n_input"]), int(dg["k"])
    L = n_input + min(k_shown, K)
    BY = {i["token"]: i for i in infos}

    frames_dir.mkdir(exist_ok=True, parents=True)
    for f_ in frames_dir.glob("frame*.png"):
        f_.unlink()

    fig = plt.figure(figsize=(11.0, 10.6), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    gsc = fig.add_gridspec(2, 1, height_ratios=(0.115, 0.62),
                           left=0.004, right=0.996, top=0.706, bottom=0.01,
                           wspace=0.06, hspace=0.10)
    ax_front = fig.add_axes((0.363, 0.760, 0.274, 0.160))
    ax_fl = fig.add_axes((0.012, 0.775, 0.2225, 0.130))
    ax_fr = fig.add_axes((0.7655, 0.775, 0.2225, 0.130))
    ax_bl = fig.add_axes((0.012, 0.553, 0.2225, 0.130))
    ax_b = fig.add_axes((0.363, 0.538, 0.274, 0.160))
    ax_br = fig.add_axes((0.7655, 0.553, 0.2225, 0.130))
    gsb = fig.add_gridspec(1, 2, left=0.055, right=0.945, top=0.492,
                           bottom=0.075, wspace=0.06)
    ax_f = fig.add_subplot(gsb[0])
    ax_r = fig.add_subplot(gsb[1])
    title = fig.suptitle("", fontsize=22, y=0.99)

    # two-row open legend below the BEV panels — shared band, paper style
    draw_legend_band(fig, 10.8, 0.030, 0.004)

    # corner sparkline: pinned into the forecast panel's upper-right corner
    ins = fig.add_axes((0.285, 0.388, 0.18, 0.092))
    ins.set_facecolor("white"); ins.patch.set_alpha(0.9)
    ins.set_xticks([]); ins.set_yticks([])
    miou_full = dg["dynamic_mious"]

    rig_labels = {"CAM_FRONT": "Front", "CAM_FRONT_LEFT": "Front Left",
                  "CAM_FRONT_RIGHT": "Front Right", "CAM_BACK": "Back",
                  "CAM_BACK_LEFT": "Back Left", "CAM_BACK_RIGHT": "Back Right"}
    CAM_ORDER = R.CAM_ORDER
    rig = [(ax_front, CAM_ORDER[0]), (ax_fl, CAM_ORDER[2]),
           (ax_fr, CAM_ORDER[1]), (ax_bl, CAM_ORDER[4]),
           (ax_b, CAM_ORDER[3]), (ax_br, CAM_ORDER[5])]

    EXT = 50.0
    for step in range(L):
        is_ctx = step < n_input
        tok = tokens[step]
        info = BY[tok]

        for ax, cam in rig:
            rel = info["cams"][cam]["data_path"]
            if rel.startswith("./data/nuscenes/"):
                rel = rel[len("./data/nuscenes/"):]
            ax.clear()
            ax.imshow(plt.imread(Path(sc["root"]) / rel))
            ax.set_title(rig_labels[cam], fontsize=13, pad=3.5, color="#333333")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(EGO_COLOR); sp.set_linewidth(1.0)

        gtb, gtn = info["gt_boxes"], info["gt_names"]
        for ax, kind in ((ax_f, "forecast"), (ax_r, "real")):
            ax.clear()
            mask = (dg["dynamic_ctx"][step] if is_ctx else
                    (dg["dynamic_pred"][step - n_input]
                     if kind == "forecast" else dg["dynamic_real"][step - n_input]))
            ax.imshow(matte_map(mask), extent=(-EXT, EXT, -EXT, EXT),
                      interpolation="bilinear", origin="upper", zorder=3)
            if kind == "forecast":
                if is_ctx:
                    draw_boxes = "ctx"
                else:
                    draw_boxes = "fc"
            else:
                draw_boxes = "me"
            if draw_boxes == "ctx":
                if not no_dets and pr is not None:
                    R.draw_boxes_bev(ax, pr["boxes_%d" % step], R.DET_COLOR, 1.9,
                                     score_thresh=SCORE, scores=pr["scores_%d" % step])
                    badge = ("observed context (input latent)\n"
                             "detections decoded · no forecast yet")
                    badge_color = R.DET_COLOR
                    ax.set_title("Map Context + Detections (Red)",
                                 fontsize=15, pad=5, color=R.DET_COLOR)
                else:
                    badge = "observed context (input latent)"
                    badge_color = EGO_COLOR
                    ax.set_title("Map Context", fontsize=15, pad=5,
                                 color=EGO_COLOR)
                ax.text(0.03, 0.035,
                        badge,
                        zorder=12, transform=ax.transAxes, fontsize=13.5, color=badge_color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#b9b9b9", lw=0.8, alpha=0.94))
            elif draw_boxes == "fc":
                kk = step - n_input + 1
                mm = float(dg["dynamic_mious"][kk - 1])
                if not no_dets and pr is not None:
                    found, n_ = R.match_gt(pr["boxes_%d" % step],
                                           pr["scores_%d" % step],
                                           pr["labels_%d" % step], gtb, gtn,
                                           score_thresh=SCORE)
                    R.draw_boxes_bev(ax, pr["boxes_%d" % step], R.DET_COLOR, 1.9,
                                     score_thresh=SCORE,
                                     scores=pr["scores_%d" % step])
                    badge = ("forecast (decoded latent)\n"
                             "mIoU %.2f · matched %d/%d GT (≤ 2 m)"
                             % (mm, found, n_))
                    badge_color = R.DET_COLOR
                    ax.set_title("Map Forecast + Detections (Red)",
                                 fontsize=15, pad=5, color=R.DET_COLOR)
                else:
                    badge = "forecast (decoded latent)\nmIoU %.2f" % mm
                    badge_color = R.DET_COLOR
                    ax.set_title("Map Forecast", fontsize=15, pad=5,
                                 color=R.DET_COLOR)
                ax.text(0.03, 0.035,
                        badge,
                        zorder=12, transform=ax.transAxes, fontsize=13.5,
                        color=badge_color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#b9b9b9", lw=0.8, alpha=0.94))
            else:
                if not no_dets and rr is not None:
                    R.draw_boxes_bev(ax, rr["boxes_%d" % step], R.ORACLE_COLOR, 1.5,
                                     score_thresh=SCORE,
                                     scores=rr["scores_%d" % step])
                R.draw_boxes_bev(ax, gtb, GT_COLOR, 1.0, dashed=True)
                ax.text(0.03, 0.035,
                        "real latent (reference)\ndecode of the measured future",
                        zorder=12, transform=ax.transAxes, fontsize=13.5,
                        color=R.ORACLE_COLOR,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#b9b9b9", lw=0.8, alpha=0.94))
                ax.set_title("Real BEV, Decoded", fontsize=15, pad=5,
                             color=R.ORACLE_COLOR)
            ax.add_patch(plt.Rectangle((-0.95, -1.9), 1.9, 4.2, fill=False,
                                       edgecolor=EGO_COLOR, lw=1.4, zorder=6))
            ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(EGO_COLOR); sp.set_linewidth(1.2)

        if is_ctx:
            ins.set_visible(False)
        else:
            kk = step - n_input + 1
            ins.set_visible(True)
            ins.clear()
            ins.set_facecolor("white"); ins.patch.set_alpha(0.9)
            ins.set_xticks([]); ins.set_yticks([])
            if agg:
                for q, st in (("naiv", "dashed"), ("ego", "dotted")):
                    c = agg[q]
                    ins.plot(np.arange(1, len(c) + 1), c,
                             color="#8a8a87", lw=1.4, ls=st)
                ins.plot(np.arange(1, len(agg["wm"]) + 1), agg["wm"],
                         color="#555555", lw=1.8)
                ins.text(0.05, 0.04, "gray = aggregate (val) · orange = scene",
                         transform=ins.transAxes, fontsize=6.2, color="#444444")
            ins.plot(np.arange(1, K + 1), miou_full, color="#B25A00", lw=2.2)
            ins.set_xlim(1, K)
            ins.set_ylim(-0.14, 1.40 if agg else
                         max(0.05, float(miou_full.max()) * 1.15))
            ins.plot([kk], [miou_full[kk - 1]], "o", color="#B25A00", ms=6,
                     mec="white", mew=0.9)
            ins.text(0.03, 0.82, "mIoU (proxy)", transform=ins.transAxes,
                     fontsize=10.0, color="#774400",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="none", alpha=0.92))
            for sp in ins.spines.values():
                sp.set_color("#b9b9b9"); sp.set_linewidth(0.8)

        title.set_text("Maps + detections rollout — nuScenes  |  "
                       "t = %.1f s" % (0.5 * (step + 1)))
        fig.savefig(frames_dir / ("frame%03d.png" % step), dpi=fig.dpi)
        print("[frames] frame%03d" % step, flush=True)
    plt.close(fig)
    return frames_dir


# ---------------------------------------------------------------------------
# artifact: GIF — assembled directly from the PNG frames (no lossy MP4 in
# the color path). Same two-pass palette as the web pipeline.
# ---------------------------------------------------------------------------
def assemble_gif(frames_dir, out_gif, width=768, fps=2):
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    pal = Path(out_gif).with_suffix(".palette.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "frame%03d.png"),
                    "-vf", "scale=%d:-2:flags=lanczos,palettegen" % width,
                    str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "frame%03d.png"),
                    "-i", str(pal),
                    "-lavfi", "scale=%d:-2:flags=lanczos [x]; "
                              "[x][1:v] paletteuse=bayer_scale=4" % width,
                    str(out_gif)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "frame%03d.png"),
                    "-vf", "scale=out_range=full",
                    "-c:v", "libx265", "-crf", "14", "-pix_fmt", "yuv444p",
                    "-color_range", "2", "-tag:v", "hvc1",
                    "-movflags", "+faststart",
                    str(Path(out_gif).with_suffix(".h265_full.mp4"))], check=True)
    print("[gif] ->", out_gif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="figure",
                    choices=["figure", "frames", "gif", "all"],
                    help="figure = paper strip; frames = rig PNG frames; "
                         "gif = frames + palette assembly (implies frames)")
    ap.add_argument("--seg-npz", required=True)
    ap.add_argument("--pred-npz", default=None,
                    help="required unless --no-dets")
    ap.add_argument("--real-npz", default=None,
                    help="required unless --no-dets")
    ap.add_argument("--no-dets", action="store_true",
                    help="map-only frames: no detection decoding / matched "
                         "counts (for windows without a proper det decode)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--agg-json", default=None,
                    help="optional decay JSON {wm:[...],naiv:[...],ego:[...]}")
    ap.add_argument("--steps", default="1,3,6",
                    help="forecast steps (k) shown as panels in the figure")
    ap.add_argument("--k-shown", type=int, default=8,
                    help="render only the first k rollout steps in the frames")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--gif-width", type=int, default=768)
    ap.add_argument("--figure-out", default="figs/qualitative_forecast.png")
    ap.add_argument("--frames-out", default="build/qualitative_frames")
    ap.add_argument("--gif-out", default=None)
    a = ap.parse_args()

    global SCORE
    SCORE = a.score_thresh

    if a.no_dets and a.artifacts in ("figure", "all"):
        sys.exit("--no-dets cannot produce the figure artifact "
                 "(the paper strip needs decoded detections)")
    if not a.no_dets and not (a.pred_npz and a.real_npz):
        sys.exit("--pred-npz/--real-npz are required unless --no-dets")

    sc = load_scene(a)
    sc["root"] = a.data_root
    steps = [int(s) for s in a.steps.split(",") if s.strip()]

    if a.artifacts in ("figure", "all"):
        render_figure(sc, steps, a.figure_out)
    if a.artifacts in ("frames", "gif", "all"):
        frames_dir = Path(a.frames_out)
        render_frames(sc, a.fps, a.k_shown, frames_dir, no_dets=a.no_dets)
        if a.artifacts in ("gif", "all"):
            out = a.gif_out or str(frames_dir.parent / "qualitative_duet.gif")
            assemble_gif(frames_dir, out, width=a.gif_width, fps=a.fps)


if __name__ == "__main__":
    main()
