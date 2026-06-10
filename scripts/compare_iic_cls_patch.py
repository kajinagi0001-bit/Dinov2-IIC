from __future__ import annotations

from pathlib import Path

import pandas as pd

from dinov2_iic.clustering import export_cluster_images


CLS_DIR = Path("outputs/iic_full_k16_e20_wandb")
PATCH_DIR = Path("outputs/iic_full_patch_mean_k16_e20_wandb")


def main() -> None:
    patch_summary = write_candidates(PATCH_DIR, "IIC Patch Mean Full K16 E20")
    cls_summary = pd.read_csv(CLS_DIR / "cluster_summary.csv")

    rows = []
    for name, summary in [
        ("cls_iic_k16", cls_summary),
        ("patch_mean_iic_k16", patch_summary),
    ]:
        nonempty = summary[summary["n_images"] > 0]
        rows.append(
            {
                "run": name,
                "used_clusters": int((summary["n_images"] > 0).sum()),
                "max_cluster_fraction": float(summary["image_fraction"].max()),
                "max_diabetes_ratio": float(nonempty["diabetes_ratio"].max()),
                "min_diabetes_ratio": float(nonempty["diabetes_ratio"].min()),
                "mean_entropy_weighted": float(
                    (nonempty["mean_entropy"] * nonempty["n_images"]).sum()
                    / nonempty["n_images"].sum()
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        "outputs/iic_cls_vs_patch_mean_k16_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=== comparison ===")
    print(comparison.to_string(index=False))
    print("=== patch candidates ===")
    print(pd.read_csv(PATCH_DIR / "candidate_clusters.csv").to_string(index=False))


def write_candidates(output_dir: Path, title: str) -> pd.DataFrame:
    assignments = pd.read_csv(output_dir / "assignments.csv")
    summary = pd.read_csv(output_dir / "cluster_summary.csv")
    export_cluster_images(assignments, output_dir / "clusters", representative_count=50)

    nonempty = summary[summary["n_images"] > 0].copy()
    rows = []
    for direction, ordered in [
        ("diabetes_enriched", nonempty.sort_values("diabetes_ratio", ascending=False)),
        ("not_diabetes_enriched", nonempty.sort_values("diabetes_ratio", ascending=True)),
    ]:
        for rank, (_, row) in enumerate(ordered.head(6).iterrows(), 1):
            cluster_id = int(row["cluster_id"])
            rows.append(
                {
                    "direction": direction,
                    "rank": rank,
                    "cluster_id": cluster_id,
                    "n_images": int(row["n_images"]),
                    "image_fraction": float(row["image_fraction"]),
                    "n_mice": int(row["n_mice"]),
                    "diabetes_ratio": float(row["diabetes_ratio"]),
                    "diabetes_images": int(row["diabetes_images"]),
                    "not_diabetes_images": int(row["not_diabetes_images"]),
                    "mean_probability": float(row["mean_probability"]),
                    "mean_entropy": float(row["mean_entropy"]),
                    "age_distribution": row["age_distribution"],
                    "thumbnail": str(
                        output_dir / "clusters" / f"cluster_{cluster_id:02d}" / "thumbnails.html"
                    ),
                }
            )

    candidates = pd.DataFrame(rows)
    candidates.to_csv(output_dir / "candidate_clusters.csv", index=False, encoding="utf-8-sig")

    lines = [
        f"# {title} Candidate Clusters",
        "",
        "| direction | rank | cluster | images | mice | diabetes_ratio | thumbnail |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidates.itertuples(index=False):
        lines.append(
            f"| {row.direction} | {row.rank} | {row.cluster_id:02d} | "
            f"{row.n_images} | {row.n_mice} | {row.diabetes_ratio:.3f} | {row.thumbnail} |"
        )
    (output_dir / "candidate_clusters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
