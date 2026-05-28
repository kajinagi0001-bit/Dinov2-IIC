# CODEX.md

## Project Status

This repository implements a DINOv2 + IIC workflow for unsupervised clustering of glomerulus images.

Current implementation covers:

- Config loading from `configs/default.yaml`
- Metadata creation from `dataset/<dataset_name>/diabetes` and `dataset/<dataset_name>/not-diabetes`
- Excel metadata join using `PAS染色標本_まとめ.xlsx`
- DINOv2 feature extraction
- k-means baseline clustering
- Cluster image export, thumbnails, metrics, and analysis report
- Frozen-DINOv2 IIC head training
- Reviewer-friendly visualization placeholders
- IIC occlusion attribution maps

## Fixed Decisions

- Standard config: `configs/default.yaml`
- Standard dataset folders: `diabetes` and `not-diabetes`
- Mouse ID parsing: extract the leading integer from the parent folder name first, then the image filename
- ID parse failure: raise an error
- Labels and age are not used for training; they are used only for post-hoc analysis
- Excel counts are authoritative: `A=122`, `WT=95`
- Label mapping: `A=diabetes`, `WT=not-diabetes`

## Main Commands

```powershell
dinov2-iic --config configs/default.yaml prepare-metadata
dinov2-iic --config configs/default.yaml extract-features
dinov2-iic --config configs/default.yaml cluster --n-clusters 8
dinov2-iic --config configs/default.yaml run-baseline --n-clusters 8
dinov2-iic --config configs/default.yaml train-iic --n-clusters 8
dinov2-iic --config configs/default.yaml assign-iic --checkpoint outputs/<experiment>/iic_head.pt
dinov2-iic --config configs/default.yaml visualize-iic --assignments outputs/<experiment>/assignments.csv --checkpoint outputs/<experiment>/iic_head.pt
```

If the package is not installed, run with:

```powershell
python -m dinov2_iic.cli --config configs/default.yaml prepare-metadata
```

## Next Work

Recommended next steps:

1. Add actual images under `dataset/mouse_dataset/diabetes` and `dataset/mouse_dataset/not-diabetes`.
2. Run `prepare-metadata` and verify the row count is close to the expected image count.
3. Run DINOv2 feature extraction on a GPU environment.
4. Run k-means for `K=4,8,16,32` and inspect `analysis_report.md`.
5. Train IIC only after the frozen-DINOv2 baseline looks biologically plausible.
6. Replace placeholder visualization with model-attribution heatmaps from the selected trained IIC head.

## Notes For Future Agents

- Do not use diabetes labels or age in training.
- Keep all splits and summaries mouse-level aware.
- Be careful with user data paths; this repo may contain only example directories.
- Avoid deleting generated outputs unless the user explicitly asks.
