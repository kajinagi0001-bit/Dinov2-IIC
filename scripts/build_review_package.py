from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Run:
    feature: str
    output_dir: Path


RUNS = [
    Run("cls", Path("outputs/full_k32")),
    Run("patch_mean", Path("outputs/full_patch_mean_k32")),
    Run("cls_patch_mean", Path("outputs/full_cls_patch_mean_k32")),
]

OUTPUT_DIR = Path("outputs/review_k32_feature_comparison")
TOP_CLUSTERS_PER_DIRECTION = 5
EXAMPLE_COUNT = 40
TOP_MOUSE_COUNT = 30


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_rows = []

    for run in RUNS:
        assignments = pd.read_csv(run.output_dir / "assignments.csv")
        summary = pd.read_csv(run.output_dir / "cluster_summary.csv")
        totals_by_mouse = assignments.groupby("mouse_id").size().rename("mouse_total_images")

        selected = select_candidate_clusters(summary)
        for _, cluster in selected.iterrows():
            cluster_id = int(cluster["cluster_id"])
            direction = cluster["direction"]
            cluster_dir = OUTPUT_DIR / run.feature / f"cluster_{cluster_id:02d}_{direction}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            cluster_assignments = assignments[assignments["cluster_id"] == cluster_id].copy()
            write_examples(cluster_assignments, cluster_dir)
            write_mouse_summary(cluster_assignments, totals_by_mouse, cluster_dir)
            write_age_label_tables(cluster_assignments, cluster_dir)

            candidate_rows.append(
                {
                    "feature": run.feature,
                    "direction": direction,
                    "cluster_id": cluster_id,
                    "n_images": int(cluster["n_images"]),
                    "n_mice": int(cluster["n_mice"]),
                    "diabetes_ratio": float(cluster["diabetes_ratio"]),
                    "diabetes_images": int(cluster["diabetes_images"]),
                    "not_diabetes_images": int(cluster["not_diabetes_images"]),
                    "age_distribution": cluster["age_distribution"],
                    "source_thumbnail": str(
                        run.output_dir / "clusters" / f"cluster_{cluster_id:02d}" / "thumbnails.html"
                    ),
                    "review_dir": str(cluster_dir),
                    "high_confidence_csv": str(cluster_dir / "high_confidence_examples.csv"),
                    "boundary_csv": str(cluster_dir / "boundary_examples.csv"),
                    "mouse_summary_csv": str(cluster_dir / "mouse_summary.csv"),
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUTPUT_DIR / "candidate_clusters.csv", index=False, encoding="utf-8-sig")
    write_markdown_index(candidates, OUTPUT_DIR / "README.md")
    print(f"Saved review package: {OUTPUT_DIR}")
    print(f"candidate clusters: {len(candidates)}")
    print(candidates[["feature", "direction", "cluster_id", "n_images", "n_mice", "diabetes_ratio"]].to_string(index=False))


def select_candidate_clusters(summary: pd.DataFrame) -> pd.DataFrame:
    diabetes = summary.sort_values("diabetes_ratio", ascending=False).head(
        TOP_CLUSTERS_PER_DIRECTION
    )
    diabetes = diabetes.assign(direction="diabetes_enriched")
    not_diabetes = summary.sort_values("diabetes_ratio", ascending=True).head(
        TOP_CLUSTERS_PER_DIRECTION
    )
    not_diabetes = not_diabetes.assign(direction="not_diabetes_enriched")
    return pd.concat([diabetes, not_diabetes], ignore_index=True)


def write_examples(cluster_assignments: pd.DataFrame, cluster_dir: Path) -> None:
    ordered = cluster_assignments.sort_values(
        ["distance_to_center", "image_path"], ascending=[True, True], kind="stable"
    )
    high_confidence = ordered.head(EXAMPLE_COUNT).copy()
    boundary = ordered.tail(EXAMPLE_COUNT).sort_values(
        ["distance_to_center", "image_path"], ascending=[False, True], kind="stable"
    )
    keep = [
        "image_path",
        "mouse_id",
        "section_id",
        "diabetes_label",
        "age_week",
        "kidney_number",
        "cluster_id",
        "cluster_probability",
        "distance_to_center",
    ]
    high_confidence[keep].to_csv(
        cluster_dir / "high_confidence_examples.csv", index=False, encoding="utf-8-sig"
    )
    boundary[keep].to_csv(cluster_dir / "boundary_examples.csv", index=False, encoding="utf-8-sig")


def write_mouse_summary(
    cluster_assignments: pd.DataFrame,
    totals_by_mouse: pd.Series,
    cluster_dir: Path,
) -> None:
    grouped = (
        cluster_assignments.groupby(["mouse_id", "diabetes_label", "age_week", "kidney_number"])
        .size()
        .rename("cluster_images")
        .reset_index()
    )
    grouped = grouped.merge(totals_by_mouse.reset_index(), on="mouse_id", how="left")
    grouped["mouse_cluster_fraction"] = grouped["cluster_images"] / grouped["mouse_total_images"]
    grouped = grouped.sort_values(
        ["cluster_images", "mouse_cluster_fraction", "mouse_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    grouped.to_csv(cluster_dir / "mouse_summary.csv", index=False, encoding="utf-8-sig")
    grouped.head(TOP_MOUSE_COUNT).to_csv(
        cluster_dir / "top_mice.csv", index=False, encoding="utf-8-sig"
    )


def write_age_label_tables(cluster_assignments: pd.DataFrame, cluster_dir: Path) -> None:
    label_age = pd.crosstab(
        cluster_assignments["age_week"],
        cluster_assignments["diabetes_label"],
        margins=True,
    )
    label_age.to_csv(cluster_dir / "age_label_table.csv", encoding="utf-8-sig")

    kidney = (
        cluster_assignments.groupby(["kidney_number", "diabetes_label"])
        .size()
        .rename("images")
        .reset_index()
        .sort_values(["images", "kidney_number"], ascending=[False, True], kind="stable")
    )
    kidney.to_csv(cluster_dir / "kidney_label_table.csv", index=False, encoding="utf-8-sig")


def write_markdown_index(candidates: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# K=32 Feature Comparison Review Package",
        "",
        "This package summarizes candidate diabetes-enriched and not-diabetes-enriched clusters across DINOv2 feature types.",
        "",
        "For each candidate cluster:",
        "",
        "- `high_confidence_examples.csv`: nearest examples to the k-means center.",
        "- `boundary_examples.csv`: farthest examples from the k-means center.",
        "- `mouse_summary.csv`: per-mouse contribution and within-mouse cluster fraction.",
        "- `top_mice.csv`: top contributing mice.",
        "- `age_label_table.csv`: age by label counts.",
        "- `kidney_label_table.csv`: kidney number by label counts.",
        "",
        "## Candidate Clusters",
        "",
        "| feature | direction | cluster | images | mice | diabetes_ratio | thumbnail | review_dir |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for _, row in candidates.iterrows():
        lines.append(
            "| {feature} | {direction} | {cluster_id:02d} | {n_images} | {n_mice} | {diabetes_ratio:.3f} | {source_thumbnail} | {review_dir} |".format(
                **row.to_dict()
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
