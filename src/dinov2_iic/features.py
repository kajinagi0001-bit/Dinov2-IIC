from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class Dinov2FeatureExtractor:
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str | None = None,
        batch_size: int = 32,
        image_size: int = 224,
        feature_type: str = "cls",
    ) -> None:
        try:
            import torch
            from torchvision import transforms
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DINOv2 feature extraction requires torch and torchvision. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.feature_type = feature_type
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def extract(self, image_paths: list[str]) -> np.ndarray:
        features: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [self._load_image(path) for path in batch_paths]
            batch = self.torch.stack(images).to(self.device)
            with self.torch.no_grad():
                feat = self._forward(batch)
            features.append(feat.detach().cpu().numpy())
        if not features:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(features, axis=0).astype(np.float32)

    def _load_image(self, path: str):
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def _forward(self, batch):
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(batch)
            if isinstance(out, dict):
                if self.feature_type == "patch_mean" and "x_norm_patchtokens" in out:
                    return out["x_norm_patchtokens"].mean(dim=1)
                if self.feature_type == "cls_patch_mean" and "x_norm_patchtokens" in out:
                    return self.torch.cat(
                        [out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)],
                        dim=1,
                    )
                if "x_norm_clstoken" in out:
                    return out["x_norm_clstoken"]
            return out
        return self.model(batch)


def extract_features(
    metadata_csv: str | Path,
    output_npz: str | Path,
    model_name: str = "dinov2_vits14",
    batch_size: int = 32,
    image_size: int = 224,
    feature_type: str = "cls",
    device: str | None = None,
) -> None:
    metadata = pd.read_csv(metadata_csv)
    extractor = Dinov2FeatureExtractor(
        model_name=model_name,
        batch_size=batch_size,
        image_size=image_size,
        feature_type=feature_type,
        device=device,
    )
    image_paths = metadata["image_path"].astype(str).tolist()
    features = extractor.extract(image_paths)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        features=features,
        image_paths=np.asarray(image_paths),
        metadata_csv=str(metadata_csv),
        model_name=model_name,
        feature_type=feature_type,
    )
