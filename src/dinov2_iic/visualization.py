from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def create_placeholder_visualizations(
    assignments_csv: str | Path,
    output_dir: str | Path,
    per_cluster: int = 20,
) -> None:
    """Create inspection overlays for clustered images.

    This is intentionally conservative: it copies the original image and adds a
    compact label banner. Model-attribution heatmaps should be generated after a
    trained IIC head is selected, but these files give reviewers a stable place
    to inspect representative examples from every cluster.
    """

    assignments = pd.read_csv(assignments_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, group in assignments.groupby("cluster_id"):
        cluster_dir = output_dir / f"cluster_{int(cluster_id):02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        ordered = group.sort_values(
            "distance_to_center" if "distance_to_center" in group.columns else "image_path",
            kind="stable",
        )
        for _, row in ordered.head(per_cluster).iterrows():
            src = Path(row["image_path"])
            if not src.exists():
                continue
            image = Image.open(src).convert("RGB")
            original_path = cluster_dir / f"mouse{int(row['mouse_id']):03d}_{src.stem}_original.jpg"
            overlay_path = cluster_dir / f"mouse{int(row['mouse_id']):03d}_{src.stem}_overlay.jpg"
            image.save(original_path, quality=95)
            _draw_banner(image, row).save(overlay_path, quality=95)


def create_iic_occlusion_visualizations(
    assignments_csv: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    per_cluster: int = 8,
    grid_size: int = 7,
    image_size: int = 224,
    device: str | None = None,
) -> None:
    try:
        import torch
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IIC attribution visualization requires torch and torchvision. Install "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc

    from .iic import ClusterHead, _dinov2_forward

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    assignments = pd.read_csv(assignments_csv)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_type = checkpoint.get("feature_type", "cls")
    encoder = torch.hub.load("facebookresearch/dinov2", checkpoint["model_name"])
    encoder.eval().to(device)
    head = ClusterHead(checkpoint["feature_dim"], checkpoint["n_clusters"]).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, group in assignments.groupby("cluster_id"):
        cluster_dir = output_dir / f"cluster_{int(cluster_id):02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        ordered = group.sort_values(
            "cluster_probability" if "cluster_probability" in group.columns else "image_path",
            ascending=False,
            kind="stable",
        )
        for _, row in ordered.head(per_cluster).iterrows():
            src = Path(row["image_path"])
            if not src.exists():
                continue
            image = Image.open(src).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            heatmap = _occlusion_heatmap(
                torch=torch,
                encoder=encoder,
                head=head,
                tensor=tensor,
                target_cluster=int(row["cluster_id"]),
                grid_size=grid_size,
                feature_type=feature_type,
            )
            stem = f"mouse{int(row['mouse_id']):03d}_{src.stem}"
            image.save(cluster_dir / f"{stem}_original.jpg", quality=95)
            save_heatmap_overlay(image, heatmap, cluster_dir / f"{stem}_heatmap.jpg")
            _draw_banner(image, row).save(cluster_dir / f"{stem}_overlay.jpg", quality=95)


def save_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    image = image.convert("RGB")
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - np.nanmin(heatmap)
    heatmap = heatmap / max(float(np.nanmax(heatmap)), 1e-8)
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size)
    heat_rgb = Image.merge("RGB", (heat, Image.new("L", heat.size), Image.new("L", heat.size)))
    overlay = Image.blend(image, heat_rgb, alpha)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, quality=95)


def _draw_banner(image: Image.Image, row: pd.Series) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    text = (
        f"cluster={int(row['cluster_id'])} mouse={int(row['mouse_id'])} "
        f"age={int(row['age_week'])} label={row['diabetes_label']}"
    )
    draw.rectangle((0, 0, out.width, 28), fill=(0, 0, 0, 170))
    draw.text((8, 7), text, fill=(255, 255, 255, 255))
    return out


def _occlusion_heatmap(
    torch,
    encoder,
    head,
    tensor,
    target_cluster: int,
    grid_size: int,
    feature_type: str = "cls",
) -> np.ndarray:
    from .iic import _dinov2_forward

    with torch.no_grad():
        base_prob = torch.softmax(head(_dinov2_forward(torch, encoder, tensor, feature_type)), dim=1)[
            0, target_cluster
        ]
    _, _, height, width = tensor.shape
    cell_h = max(1, height // grid_size)
    cell_w = max(1, width // grid_size)
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    occluded_batches = []
    coords = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = gy * cell_h
            x0 = gx * cell_w
            y1 = height if gy == grid_size - 1 else min(height, y0 + cell_h)
            x1 = width if gx == grid_size - 1 else min(width, x0 + cell_w)
            occluded = tensor.clone()
            occluded[:, :, y0:y1, x0:x1] = 0.0
            occluded_batches.append(occluded)
            coords.append((gy, gx))
    batch = torch.cat(occluded_batches, dim=0)
    with torch.no_grad():
        probs = torch.softmax(head(_dinov2_forward(torch, encoder, batch, feature_type)), dim=1)[
            :, target_cluster
        ]
    drops = (base_prob - probs).detach().cpu().numpy()
    for (gy, gx), drop in zip(coords, drops):
        heatmap[gy, gx] = max(float(drop), 0.0)
    return heatmap
