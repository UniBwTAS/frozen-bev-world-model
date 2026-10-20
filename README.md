# BEV World Model — Forecasting Fused BEV Latents (nuScenes)

<p align="center">
  <a href="checkpoints/README.md"><img src="https://img.shields.io/badge/%E2%AC%87_Weights-seg_%2B_det_checkpoints_(2%C3%9773_MB)-blue" alt="Download the trained world-model checkpoints"></a>
  <a href="https://anonymous.4open.science/w/9b2A/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Project_page-%20frozen--bev--world--model-green" alt="Open the interactive project page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT license"></a>
</p>

A compact Transformer forecasts the next fused bird's-eye-view (BEV) latent
of a frozen BEVFusion perception stack 0.5 s ahead from three preceding
latents; the original frozen segmentation and detection heads decode the
forecast without retraining the perception system.

This is the **public companion repository** of the paper
*"Extending BEV Perception into the Future: Reusing Frozen Perception Stack"*
(preprint; the venue will be stated here once the paper is accepted). It
provides the demo media,
the self-contained demo pipeline, and the full source code with an
English packaging (semantics unchanged).

## Demo

![Busy intersection rollout: forecast maps + detections](img/forecast_f1b7b463.gif)

Duets of six nuScenes cameras (2 Hz) and the decoded BEV grids. Every
colored element in the duet GIFs is decoded — maps (left) = the world
model's map forecast, red boxes = detections decoded from the same
forecast latent; the right panel is always the measured reference:
cyan = detections decoded from the real future latent, gray dashes =
actual traffic (GT). Each frame carries the per-step proxy mIoU
(forecast decode vs. decoded real latent — no annotations needed) and the
matched GT-vehicle count; the corner inset overlays the scene's own mIoU
decay on the aggregate validation curves (solid = forecast, dashed =
naive persistence, dotted = ego-compensated persistence). Scene selection
is disclosed by computed statistics (drift ratio / front-arc traffic
count / gate-α extremes) — no image-quality cherry-picking. The GIFs
cover the first 4 s of a k = 21 rollout.

| Asset | Scene | Notes |
|---|---|---|
| `img/forecast_d5324133.gif` | construction street (drift 1.38) | maps + detections (red); matched up to 13/17 |
| `img/forecast_f1b7b463.gif` | busy intersection (drift 0.98) | maps + detections |
| `img/forecast_029af235.gif` | fast street (drift 1.72) | maps + detections, speed-extrapolation limit |
| `img/forecast_gatealpha.gif` | gate-α maximum (ᾱ 0.71) | copy/predict predict pole; moving map |
| `img/forecast_crossing.gif` | gate-α minimum (ᾱ 0.03) | copy pole; matched up to 47/64 |
| `docs/assets/hero_combo_3e8750f3.gif` | downtown street (ᾱ 0.44, drift 0.46) | duet (webpage Plate VI); mIoU 0.84 → 0.29 in 4 s; matched up to 17/36 |
| `docs/assets/hero_combo_134c7f63.gif` | roadworks street (ᾱ 0.57, drift 0.13) | duet (webpage Plate VII); tightest motion tracking; mIoU 0.76 → 0.52 in 4 s; matched up to 9/16 |
| `img/maps_gate_pure.gif` / `img/maps_crossing_pure.gif` | gate | pure map-only variants (no boxes) |
| `img/rollout_filmstrip.png` | dynamic | paper-style filmstrip (context + k = 1..4) |

## Headline results (full validation, against nuScenes annotations)

| Metric | Persistence | World model | Real latent (reference) |
|---|---|---|---|
| Map segmentation (mIoU, 6 classes) | 0.4647 | **0.5898** | 0.6295 |
| 3D detection (mAP, energy + cosine) | 0.1696 | **0.3752** | 0.6858 |

- ~6 M parameters, 4.4–11.5 ms per forecast step (TITAN RTX), plus ~173 ms
  for the separately executed BEVFusion encoder.
- Task-dependent objectives: dense segmentation needs little beyond
  Smooth-L1 reconstruction + channel-spread matching; sparse detection
  additionally benefits from energy weighting and cosine alignment.
- Autoregressive rollout beats persistence at every step up to k = 10 (5 s).

## Repository structure

Flat library layout: all model modules live in the repository root.

```
├── train_linux.py            # training conductor (locally + cluster)
├── inference.py              # proxy evaluation seg (300-window subset)
├── eval_full_val.py          # full-validation seg (5,743 windows)
├── rollout_eval.py           # autoregressive rollout evaluation
├── bev_dataset.py            # latent dataset (windows, packed memmap, HDF5)
├── bev_dataloader.py         # dataloader factory (multiple PKL sources)
├── bev_world_model.py        # the world model (transformer + gate + head)
├── transformer.py            # Pre-LN transformer blocks, embeddings
├── embedding.py, output_head.py, upsampling_head.py, flow_head.py
├── seg_decoder_torch.py      # frozen BEVFusion seg decoder + IoU utilities
├── config.py, downsampling.py, checkpointing.py
├── pack_latents.py, build_hdf5_cache.py, shm_staging.py, test_output.py
├── tools/                    # head adaptation: adapt_seg_head.py (decoder,
│                             #   seg) and adapt_det_head.py (TransFusion, det)
├── bevfusion_patch/          # latent extraction/injection: latent_saver.py
│                             #   + hook instructions (SAVE_/LOAD_BEV_LATENTS)
├── configs/                  # example configs (best operating points seg/det)
├── scripts/                  # demo scripts (rollout compute + renders)
├── img/                      # demo media (hero videos/GIFs, filmstrip)
├── environment.yml           # conda reference environment (training/inference)
└── requirements.txt          # the same environment as a pip variant
```

Comments, docstrings, and console output are English (repackaged from the
original codebase); markdown tables and code semantics are unchanged.

## Setup

```bash
# conda variant (recommended):
conda env create -f environment.yml && conda activate bevwm
# pip variant (Python 3.10, CUDA 12.1):
pip install -r requirements.txt
```

## Reproducing the demos

`scripts/` reproduces the rollout demos; they import the model, dataset,
and frozen decoder from this repository (`train_linux.py`, `Code/`):

```bash
# 1. Rollout + decoding + gate-alpha scene selection -> NPZ
python3 scripts/rollout_compute.py \
    --data-root  path/to/nuscenes/complete \
    --pkgs-root  path/to/nuscenes/latents_packed_fp16 \
    --checkpoint checkpoints/00_BEST/seg/best_miou.pt \
    --seg-head   path/to/bevfusion-seg.pth \
    --k 21 --out rollout.npz

# 2a. Paper-style filmstrip (PNG/PDF)
python3 scripts/render_filmstrip.py --npz rollout.npz --out img

# 2b. Hero video (6 cameras + LiDAR-BEV duett, MP4)
python3 scripts/render_hero.py --npz rollout.npz \
    --data-root path/to/nuscenes/complete --out img
```

What is **not** in this repository (license): nuScenes data, BEVFusion
weights (`bevfusion-seg.pth` / `bevfusion-det.pth`, official
[mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) releases),
the adapted-head variants of the head-adaptation diagnostic, the trained
world-model checkpoint files (released after acceptance — see
[checkpoints/README.md](checkpoints/README.md)), and packed
latents. Latents are extracted
with the `latent_saver` hook (`SAVE_BEV_LATENTS`, documented under
`bevfusion_patch/` in this repository) and packed with
`Code/pack_latents.py`; annotation-metric evaluation runs via latent
injection in the BEVFusion container (`LOAD_BEV_LATENTS`). The
segmentation demo pipeline above runs in plain PyTorch on a single GPU.

## Citation

Paper reference and BibTeX will be added here at submission; the code
artifact is citable as:

> **frozen-bev-world-model: Forecasting fused BEV latents on a frozen
> BEVFusion perception stack.** L. Pohl, V. Mann, 2026.
> github.com/UniBwTAS/frozen-bev-world-model

This work builds on the institute's open codebase
([github.com/Heeresfliegermogli/Masterarbeit-BEV-WorldModel](https://github.com/Heeresfliegermogli/Masterarbeit-BEV-WorldModel)).

## License

MIT — see [LICENSE](LICENSE). nuScenes data and BEVFusion weights remain
under their respective providers' terms and are not part of this repository.
