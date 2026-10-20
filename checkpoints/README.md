# Trained world-model checkpoints

The two final operating points of the latent forecaster — the exact
weights behind the paper's numbers.

**Release policy:** the checkpoint files (73 MB each) are published with
the camera-ready version, after acceptance. During double-blind review
they cannot be shipped through the anonymized mirror: the anonymous
hosting pipeline does not support files of this size, and a
non-anonymized download link would break the review's anonymization.
The measured evaluation records below are included instead, so the
reported numbers stay inspectable during review.

| Checkpoint | Task / objective | Reported quality |
|---|---|---|
| `seg/best_miou.pt` | map segmentation, Smooth-L1 + channel-spread | proxy mIoU **0.6946** (5,743 windows) · annotation mIoU **0.5898** (full validation) |
| `det/best_val_loss.pt` | 3D detection, Smooth-L1 + spread + energy (α=4) + cosine (0.1) | annotation mAP **0.3752** (mean over two seeds) |

Reference configs: `configs/config_seg_beispiel.yaml` and
`configs/config_det_beispiel.yaml` (best operating points).

Measured runs (included): [`eval/miou_fullval.json`](eval/miou_fullval.json),
[`eval/det_eval_d5ac_energycos.json`](eval/det_eval_d5ac_energycos.json),
[`eval/det_eval_d5ac_s43.json`](eval/det_eval_d5ac_s43.json).

## Checksums (of the files to be released)

```
38db8063f59788197afa503f965c8dda  seg/best_miou.pt
935fec0a838a00448ac94347b5499815  det/best_val_loss.pt
```

## Loading (once released)

`train_linux.py` checkpoint format:

```bash
# segmentation rollout / demo pipeline (plain PyTorch, single GPU)
python3 rollout_eval.py --checkpoint checkpoints/seg/best_miou.pt \
    --config configs/config_seg_beispiel.yaml

# detection: dump forecast latents and decode via latent injection
# (LOAD_BEV_LATENTS) in the BEVFusion environment — see bevfusion_patch/
```

The segmentation demo pipeline (`scripts/`) runs end-to-end in plain
PyTorch; nuScenes data, packed latents, and the frozen BEVFusion
weights are **not** part of this repository (license) — see the top-level
README for what is excluded and how latents are extracted.

## Honesty notes

- **Proxy vs. annotation metrics are different families** — the proxy
  mIoU compares the forecast decode against the decoded real future
  latent (no labels); annotation metrics run the injection pipeline
  against nuScenes labels. Never compare the two.
- The segmentation checkpoint is the thesis operating point; the
  detection checkpoint is the post-thesis energy-plus-cosine
  configuration used for the paper's detection numbers.
- The released files are the exact artifacts from training (no
  post-processing); single-run results are subject to the seed
  variation band reported in the paper.

## License

Upon release, the checkpoints carry the repository's MIT license (own
trained weights).
