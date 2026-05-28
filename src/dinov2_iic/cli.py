from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from .clustering import run_kmeans
from .config import load_config
from .features import extract_features
from .iic import assign_iic, train_iic
from .metadata import build_metadata, save_metadata
from .visualization import create_iic_occlusion_visualizations, create_placeholder_visualizations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dinov2-iic")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("prepare-metadata", help="Scan dataset and create metadata.csv")
    p.add_argument("--output", default=None)

    p = subparsers.add_parser("extract-features", help="Extract frozen DINOv2 features")
    p.add_argument("--metadata", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--feature-type", default=None)
    p.add_argument("--device", default=None)

    p = subparsers.add_parser("cluster", help="Run k-means and export clustered images")
    p.add_argument("--metadata", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-clusters", type=int, required=True)
    p.add_argument("--representative-count", type=int, default=30)
    p.add_argument("--no-copy-images", action="store_true")

    p = subparsers.add_parser("train-iic", help="Train frozen-DINOv2 IIC cluster head")
    p.add_argument("--metadata", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-clusters", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", default=None)

    p = subparsers.add_parser("assign-iic", help="Assign images with a trained IIC head")
    p.add_argument("--metadata", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--representative-count", type=int, default=30)
    p.add_argument("--no-copy-images", action="store_true")

    p = subparsers.add_parser("visualize", help="Create reviewer-friendly visualization files")
    p.add_argument("--assignments", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--per-cluster", type=int, default=20)

    p = subparsers.add_parser("visualize-iic", help="Create occlusion attribution maps for IIC")
    p.add_argument("--assignments", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--per-cluster", type=int, default=8)
    p.add_argument("--grid-size", type=int, default=7)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", default=None)

    p = subparsers.add_parser("run-baseline", help="Prepare metadata, extract features, and cluster")
    p.add_argument("--n-clusters", type=int, default=None)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "prepare-metadata":
        output = Path(args.output or _path(cfg, "metadata.output_csv", "outputs/metadata.csv"))
        df = build_metadata(cfg)
        save_metadata(df, output)
        print(f"Saved metadata: {output} ({len(df)} images)")
        return

    if args.command == "extract-features":
        feature_cfg = cfg.get("features", {})
        metadata = args.metadata or _path(cfg, "metadata.output_csv", "outputs/metadata.csv")
        output = args.output or _path(cfg, "features.output_npz", "outputs/features_dinov2.npz")
        extract_features(
            metadata_csv=metadata,
            output_npz=output,
            model_name=args.model_name or feature_cfg.get("model_name", "dinov2_vits14"),
            batch_size=args.batch_size or feature_cfg.get("batch_size", 32),
            image_size=args.image_size or feature_cfg.get("image_size", 224),
            feature_type=args.feature_type or feature_cfg.get("feature_type", "cls"),
            device=args.device,
        )
        print(f"Saved features: {output}")
        return

    if args.command == "cluster":
        experiment = cfg.get("experiment", {})
        metadata = args.metadata or _path(cfg, "metadata.output_csv", "outputs/metadata.csv")
        features = args.features or _path(cfg, "features.output_npz", "outputs/features_dinov2.npz")
        output_dir = Path(args.output_dir or _experiment_dir(cfg, f"kmeans_k{args.n_clusters}"))
        run_kmeans(
            metadata_csv=metadata,
            features_npz=features,
            output_dir=output_dir,
            n_clusters=args.n_clusters,
            seed=experiment.get("seed", 42),
            copy_images=not args.no_copy_images,
            representative_count=args.representative_count,
        )
        _copy_config(args.config, output_dir)
        print(f"Saved cluster outputs: {output_dir}")
        return

    if args.command == "train-iic":
        iic_cfg = cfg.get("iic", {})
        metadata = args.metadata or _path(cfg, "metadata.output_csv", "outputs/metadata.csv")
        output_dir = Path(args.output_dir or _experiment_dir(cfg, f"iic_k{args.n_clusters}"))
        train_iic(
            metadata_csv=metadata,
            output_dir=output_dir,
            n_clusters=args.n_clusters,
            epochs=args.epochs or iic_cfg.get("epochs", 20),
            batch_size=args.batch_size or iic_cfg.get("batch_size", 32),
            lr=args.lr or iic_cfg.get("lr", 1e-3),
            model_name=args.model_name or cfg.get("features", {}).get("model_name", "dinov2_vits14"),
            image_size=args.image_size or cfg.get("features", {}).get("image_size", 224),
            device=args.device,
        )
        _copy_config(args.config, output_dir)
        print(f"Saved IIC model outputs: {output_dir}")
        return

    if args.command == "assign-iic":
        metadata = args.metadata or _path(cfg, "metadata.output_csv", "outputs/metadata.csv")
        output_dir = Path(args.output_dir or _experiment_dir(cfg, "iic_assignments"))
        assign_iic(
            metadata_csv=metadata,
            checkpoint_path=args.checkpoint,
            output_dir=output_dir,
            batch_size=args.batch_size or cfg.get("iic", {}).get("batch_size", 32),
            image_size=args.image_size or cfg.get("features", {}).get("image_size", 224),
            device=args.device,
            copy_images=not args.no_copy_images,
            representative_count=args.representative_count,
        )
        _copy_config(args.config, output_dir)
        print(f"Saved IIC assignments: {output_dir}")
        return

    if args.command == "visualize":
        output_dir = Path(args.output_dir or Path(args.assignments).parent / "visualizations")
        create_placeholder_visualizations(args.assignments, output_dir, args.per_cluster)
        print(f"Saved visualizations: {output_dir}")
        return

    if args.command == "visualize-iic":
        output_dir = Path(args.output_dir or Path(args.assignments).parent / "visualizations_iic")
        create_iic_occlusion_visualizations(
            assignments_csv=args.assignments,
            checkpoint_path=args.checkpoint,
            output_dir=output_dir,
            per_cluster=args.per_cluster,
            grid_size=args.grid_size,
            image_size=args.image_size or cfg.get("features", {}).get("image_size", 224),
            device=args.device,
        )
        print(f"Saved IIC attribution visualizations: {output_dir}")
        return

    if args.command == "run-baseline":
        n_clusters = args.n_clusters or cfg.get("experiment", {}).get("cluster_counts", [8])[0]
        experiment_name = args.experiment_name or _timestamped_name(f"baseline_k{n_clusters}")
        output_root = Path(cfg.get("experiment", {}).get("output_dir", "outputs")) / experiment_name
        metadata_csv = output_root / "metadata.csv"
        features_npz = output_root / "features_dinov2.npz"
        df = build_metadata(cfg)
        save_metadata(df, metadata_csv)
        extract_features(
            metadata_csv=metadata_csv,
            output_npz=features_npz,
            model_name=cfg.get("features", {}).get("model_name", "dinov2_vits14"),
            batch_size=cfg.get("features", {}).get("batch_size", 32),
            image_size=cfg.get("features", {}).get("image_size", 224),
            feature_type=cfg.get("features", {}).get("feature_type", "cls"),
            device=args.device,
        )
        run_kmeans(
            metadata_csv=metadata_csv,
            features_npz=features_npz,
            output_dir=output_root,
            n_clusters=n_clusters,
            seed=cfg.get("experiment", {}).get("seed", 42),
            copy_images=True,
        )
        _copy_config(args.config, output_root)
        print(f"Saved baseline experiment: {output_root}")


def _path(cfg: dict, dotted: str, default: str) -> str:
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return str(cur)


def _experiment_dir(cfg: dict, name: str) -> Path:
    return Path(cfg.get("experiment", {}).get("output_dir", "outputs")) / _timestamped_name(name)


def _timestamped_name(prefix: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prefix}"


def _copy_config(config_path: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")


if __name__ == "__main__":
    main()
