from __future__ import annotations

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
) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IIC training requires torch and torchvision. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(metadata_csv)

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
    loader = DataLoader(
        TwoViewDataset(metadata["image_path"].astype(str).tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    log_rows = []
    for epoch in range(1, epochs + 1):
        losses = []
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
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(1, len(losses))
        log_rows.append({"epoch": epoch, "loss": mean_loss})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "n_clusters": n_clusters,
            "feature_dim": feature_dim,
            "head_state_dict": head.state_dict(),
        },
        output_dir / "iic_head.pt",
    )
    pd.DataFrame(log_rows).to_csv(output_dir / "train_log.csv", index=False)


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
    if copy_images:
        export_cluster_images(assignments, output_dir / "clusters", representative_count)


class ClusterHead:
    def __new__(cls, feature_dim: int, n_clusters: int):
        import torch
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
