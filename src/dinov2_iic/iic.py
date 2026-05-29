from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def train_iic(
    metadata_csv: str | Path,
    output_dir: str | Path,
    n_clusters: int,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    model_name: str = "dinov2_vits14",
    image_size: int = 224,
    device: str | None = None,
    seed: int = 42,
    assign_after_train: bool = True,
    copy_images: bool = True,
    representative_count: int = 30,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "disabled",
    wandb_tags: list[str] | None = None,
) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IIC training requires torch and torchvision. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = Path(metadata_csv)
    metadata = pd.read_csv(metadata_csv)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(seed, torch)

    run_config = {
        "metadata_csv": str(metadata_csv),
        "output_dir": str(output_dir),
        "n_images": int(len(metadata)),
        "n_clusters": int(n_clusters),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "model_name": model_name,
        "image_size": int(image_size),
        "device": device,
        "seed": int(seed),
        "assign_after_train": bool(assign_after_train),
        "copy_images": bool(copy_images),
        "representative_count": int(representative_count),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    wandb_run = _init_wandb(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        mode=wandb_mode,
        tags=wandb_tags,
        config=run_config,
    )

    base_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    class TwoViewDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            image = Image.open(self.paths[idx]).convert("RGB")
            return base_transform(image), base_transform(image)

    encoder = torch.hub.load("facebookresearch/dinov2", model_name)
    encoder.eval().to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)

    feature_dim = _infer_feature_dim(torch, encoder, device, image_size)
    head = ClusterHead(feature_dim, n_clusters).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TwoViewDataset(metadata["image_path"].astype(str).tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )

    log_rows = []
    usage_rows = []
    for epoch in range(1, epochs + 1):
        losses = []
        epoch_counts = torch.zeros(n_clusters, device=device)
        epoch_prob_sum = torch.zeros(n_clusters, device=device)
        n_seen = 0
        entropy_values = []
        for view_a, view_b in loader:
            view_a = view_a.to(device)
            view_b = view_b.to(device)
            with torch.no_grad():
                feat_a = _dinov2_forward(torch, encoder, view_a)
                feat_b = _dinov2_forward(torch, encoder, view_b)
            prob_a = torch.softmax(head(feat_a), dim=1)
            prob_b = torch.softmax(head(feat_b), dim=1)
            loss = iic_loss(prob_a, prob_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.cat([prob_a.detach(), prob_b.detach()], dim=0)
            assignments = probs.argmax(dim=1)
            epoch_counts += torch.bincount(assignments, minlength=n_clusters).float()
            epoch_prob_sum += probs.sum(dim=0)
            n_seen += int(probs.shape[0])
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            entropy_values.append(entropy.detach())
            losses.append(float(loss.detach().cpu()))

        mean_loss = sum(losses) / max(1, len(losses))
        counts_np = epoch_counts.detach().cpu().numpy()
        fractions = counts_np / max(1, counts_np.sum())
        mean_probs = (epoch_prob_sum / max(1, n_seen)).detach().cpu().numpy()
        entropies = torch.cat(entropy_values).detach().cpu().numpy() if entropy_values else np.asarray([])
        used_clusters = int((counts_np > 0).sum())
        max_cluster_fraction = float(fractions.max()) if len(fractions) else 0.0
        mean_entropy = float(entropies.mean()) if len(entropies) else float("nan")

        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "used_clusters": used_clusters,
            "max_cluster_fraction": max_cluster_fraction,
            "mean_entropy": mean_entropy,
            "n_views": int(n_seen),
        }
        log_rows.append(row)
        for cluster_id, (count, fraction, mean_prob) in enumerate(
            zip(counts_np, fractions, mean_probs)
        ):
            usage_rows.append(
                {
                    "epoch": epoch,
                    "cluster_id": cluster_id,
                    "assigned_views": int(count),
                    "assigned_fraction": float(fraction),
                    "mean_probability": float(mean_prob),
                }
            )

        _wandb_log(wandb_run, row)
        print(
            f"epoch={epoch} loss={mean_loss:.6f} "
            f"used_clusters={used_clusters}/{n_clusters} "
            f"max_cluster_fraction={max_cluster_fraction:.3f} "
            f"mean_entropy={mean_entropy:.3f}"
        )

    torch.save(
        {
            "model_name": model_name,
            "n_clusters": n_clusters,
            "feature_dim": feature_dim,
            "image_size": image_size,
            "seed": seed,
            "head_state_dict": head.state_dict(),
            "run_config": run_config,
        },
        output_dir / "iic_head.pt",
    )
    train_log = pd.DataFrame(log_rows)
    usage = pd.DataFrame(usage_rows)
    train_log.to_csv(output_dir / "train_log.csv", index=False)
    usage.to_csv(output_dir / "cluster_usage_by_epoch.csv", index=False)
    _write_training_report(train_log, usage, output_dir / "training_report.md", n_clusters)

    if assign_after_train:
        assign_iic(
            metadata_csv=metadata_csv,
            checkpoint_path=output_dir / "iic_head.pt",
            output_dir=output_dir,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            copy_images=copy_images,
            representative_count=representative_count,
        )
        final_summary = pd.read_csv(output_dir / "cluster_summary.csv")
        _wandb_log(
            wandb_run,
            {
                "final/used_clusters": int((final_summary["n_images"] > 0).sum()),
                "final/max_cluster_fraction": float(
                    final_summary["n_images"].max() / final_summary["n_images"].sum()
                ),
                "final/max_diabetes_ratio": float(final_summary["diabetes_ratio"].max()),
                "final/min_diabetes_ratio": float(final_summary["diabetes_ratio"].min()),
            },
        )

    if wandb_run is not None:
        wandb_run.finish()


def assign_iic(
    metadata_csv: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    device: str | None = None,
    copy_images: bool = True,
    representative_count: int = 30,
) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IIC assignment requires torch and torchvision. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    from PIL import Image
    from .clustering import export_cluster_images

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(metadata_csv)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    class ImageDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            image = Image.open(self.paths[idx]).convert("RGB")
            return transform(image)

    loader = DataLoader(
        ImageDataset(metadata["image_path"].astype(str).tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features = _dinov2_forward(torch, encoder, batch)
            probs = torch.softmax(head(features), dim=1)
            all_probs.append(probs.cpu().numpy())
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0))
    cluster_id = probs_np.argmax(axis=1) if len(probs_np) else np.asarray([], dtype=int)
    max_prob = probs_np.max(axis=1) if len(probs_np) else np.asarray([], dtype=float)
    entropy = -(probs_np * np.log(probs_np + 1e-8)).sum(axis=1) if len(probs_np) else np.asarray([])

    assignments = metadata.copy()
    assignments["cluster_id"] = cluster_id
    assignments["cluster_probability"] = max_prob
    assignments["entropy"] = entropy
    assignments["distance_to_center"] = 1.0 - max_prob

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "assignments.csv", index=False, encoding="utf-8-sig")
    n_clusters = int(checkpoint["n_clusters"])
    _write_cluster_summary(assignments, output_dir / "cluster_summary.csv", n_clusters)
    _write_assignment_report(assignments, output_dir / "analysis_report.md", n_clusters)
    if copy_images:
        export_cluster_images(assignments, output_dir / "clusters", representative_count)


class ClusterHead:
    def __new__(cls, feature_dim: int, n_clusters: int):
        from torch import nn

        return nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_clusters),
        )


def iic_loss(prob_a, prob_b, eps: float = 1e-8):
    torch = __import__("torch")
    joint = prob_a.unsqueeze(2) * prob_b.unsqueeze(1)
    joint = joint.mean(dim=0)
    joint = (joint + joint.t()) / 2.0
    joint = joint / joint.sum()
    pi = joint.sum(dim=1, keepdim=True)
    pj = joint.sum(dim=0, keepdim=True)
    loss = -joint * (torch.log(joint + eps) - torch.log(pi + eps) - torch.log(pj + eps))
    return loss.sum()


def _set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_feature_dim(torch, encoder, device: str, image_size: int) -> int:
    dummy = torch.zeros(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        feat = _dinov2_forward(torch, encoder, dummy)
    return int(feat.shape[1])


def _dinov2_forward(torch, encoder, batch):
    if hasattr(encoder, "forward_features"):
        out = encoder.forward_features(batch)
        if isinstance(out, dict) and "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
    out = encoder(batch)
    if out.ndim > 2:
        out = out.flatten(1)
    return out


def _write_cluster_summary(assignments: pd.DataFrame, output_path: Path, n_clusters: int) -> None:
    rows = []
    total = max(1, len(assignments))
    grouped = {int(cluster_id): group for cluster_id, group in assignments.groupby("cluster_id")}
    for cluster_id in range(n_clusters):
        group = grouped.get(cluster_id)
        if group is None or group.empty:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "n_images": 0,
                    "image_fraction": 0.0,
                    "n_mice": 0,
                    "diabetes_images": 0,
                    "not_diabetes_images": 0,
                    "diabetes_ratio": np.nan,
                    "mean_probability": np.nan,
                    "mean_entropy": np.nan,
                    "age_distribution": {},
                }
            )
            continue
        label_counts = group["diabetes_label"].value_counts().to_dict()
        age_counts = group["age_week"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_images": int(len(group)),
                "image_fraction": len(group) / total,
                "n_mice": int(group["mouse_id"].nunique()),
                "diabetes_images": int(label_counts.get("diabetes", 0)),
                "not_diabetes_images": int(label_counts.get("not-diabetes", 0)),
                "diabetes_ratio": label_counts.get("diabetes", 0) / len(group),
                "mean_probability": float(group["cluster_probability"].mean()),
                "mean_entropy": float(group["entropy"].mean()),
                "age_distribution": age_counts,
            }
        )
    pd.DataFrame(rows).sort_values("cluster_id").to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def _write_assignment_report(assignments: pd.DataFrame, output_path: Path, n_clusters: int) -> None:
    summary_rows = []
    used_clusters = int(assignments["cluster_id"].nunique())
    cluster_sizes = assignments["cluster_id"].value_counts()
    max_cluster_fraction = float(cluster_sizes.max() / len(assignments)) if len(assignments) else 0.0
    collapse_warning = used_clusters < max(2, n_clusters // 2) or max_cluster_fraction > 0.5
    for cluster_id, group in assignments.groupby("cluster_id"):
        labels = group["diabetes_label"].value_counts().to_dict()
        ages = group["age_week"].value_counts().sort_index().to_dict()
        summary_rows.extend(
            [
                f"### Cluster {int(cluster_id):02d}",
                "",
                f"- images: {len(group)}",
                f"- mice: {group['mouse_id'].nunique()}",
                f"- diabetes_ratio: {labels.get('diabetes', 0) / len(group):.4f}",
                f"- labels: {labels}",
                f"- ages: {ages}",
                f"- mean_probability: {group['cluster_probability'].mean():.4f}",
                f"- mean_entropy: {group['entropy'].mean():.4f}",
                "",
            ]
        )
    output_path.write_text(
        "\n".join(
            [
                "# IIC Assignment Analysis",
                "",
                "## Collapse Check",
                "",
                f"- used_clusters: {used_clusters} / {n_clusters}",
                f"- max_cluster_fraction: {max_cluster_fraction:.4f}",
                f"- warning: {collapse_warning}",
                "",
                "## Cluster Summary",
                "",
                *summary_rows,
            ]
        ),
        encoding="utf-8",
    )


def _write_training_report(
    train_log: pd.DataFrame,
    usage: pd.DataFrame,
    output_path: Path,
    n_clusters: int,
) -> None:
    last = train_log.iloc[-1].to_dict() if not train_log.empty else {}
    lines = [
        "# IIC Training Report",
        "",
        "## Final Epoch",
        "",
        f"- loss: {last.get('loss', float('nan'))}",
        f"- used_clusters: {last.get('used_clusters', 0)} / {n_clusters}",
        f"- max_cluster_fraction: {last.get('max_cluster_fraction', float('nan'))}",
        f"- mean_entropy: {last.get('mean_entropy', float('nan'))}",
        "",
        "## Collapse Check",
        "",
    ]
    max_fraction = float(last.get("max_cluster_fraction", 1.0))
    used_clusters = int(last.get("used_clusters", 0))
    if used_clusters < max(2, n_clusters // 2) or max_fraction > 0.5:
        lines.append("Potential collapse warning: cluster usage is highly imbalanced.")
    else:
        lines.append("No obvious collapse warning from epoch-level cluster usage.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _init_wandb(
    project: str | None,
    entity: str | None,
    name: str | None,
    mode: str,
    tags: list[str] | None,
    config: dict,
):
    enabled = mode != "disabled" or project is not None
    if not enabled:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "W&B logging was requested but wandb is not installed. "
            "Install it with `pip install wandb` or use `--wandb-mode disabled`."
        ) from exc
    actual_mode = "online" if mode == "disabled" else mode
    return wandb.init(
        project=project or "dinov2-iic",
        entity=entity,
        name=name,
        mode=actual_mode,
        tags=tags,
        config=config,
    )


def _wandb_log(run, row: dict) -> None:
    if run is not None:
        run.log(row)
