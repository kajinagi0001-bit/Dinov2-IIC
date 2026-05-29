from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


RUNS = {
    "cls": Path("outputs/full_k32"),
    "patch_mean": Path("outputs/full_patch_mean_k32"),
    "cls_patch_mean": Path("outputs/full_cls_patch_mean_k32"),
}


def main() -> None:
    comparison_rows = []
    cluster_rows = []
    assignments = {}

    for name, output_dir in RUNS.items():
        metrics = pd.read_csv(output_dir / "metrics.csv").iloc[0].to_dict()
        summary = pd.read_csv(output_dir / "cluster_summary.csv")
        assignment = pd.read_csv(output_dir / "assignments.csv")
        assignments[name] = assignment

        comparison_rows.append(
            {
                "feature": name,
                "n_images": int(metrics["n_images"]),
                "silhouette": float(metrics["silhouette_score"]),
                "davies_bouldin": float(metrics["davies_bouldin_index"]),
                "min_cluster_images": int(summary["n_images"].min()),
                "max_cluster_images": int(summary["n_images"].max()),
                "min_cluster_mice": int(summary["n_mice"].min()),
                "max_diabetes_ratio": float(summary["diabetes_ratio"].max()),
                "min_diabetes_ratio": float(summary["diabetes_ratio"].min()),
            }
        )

        add_candidates(
            cluster_rows,
            feature=name,
            output_dir=output_dir,
            summary=summary,
            direction="diabetes_enriched",
            ascending=False,
        )
        add_candidates(
            cluster_rows,
            feature=name,
            output_dir=output_dir,
            summary=summary,
            direction="not_diabetes_enriched",
            ascending=True,
        )

    agreement_rows = []
    names = list(RUNS)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left = assignments[left_name]["cluster_id"].to_numpy()
            right = assignments[right_name]["cluster_id"].to_numpy()
            agreement_rows.append(
                {
                    "feature_a": left_name,
                    "feature_b": right_name,
                    "adjusted_rand_index": adjusted_rand_score(left, right),
                    "normalized_mutual_info": normalized_mutual_info_score(left, right),
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    clusters = pd.DataFrame(cluster_rows)
    agreement = pd.DataFrame(agreement_rows)

    comparison.to_csv("outputs/feature_type_k32_comparison.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(
        "outputs/feature_type_k32_candidate_clusters.csv", index=False, encoding="utf-8-sig"
    )
    agreement.to_csv(
        "outputs/feature_type_k32_assignment_agreement.csv", index=False, encoding="utf-8-sig"
    )

    print("=== feature comparison ===")
    print(comparison.to_string(index=False))
    print("\n=== assignment agreement ===")
    print(agreement.to_string(index=False))
    print("\n=== candidate clusters ===")
    print(clusters.to_string(index=False))


def add_candidates(
    rows: list[dict],
    feature: str,
    output_dir: Path,
    summary: pd.DataFrame,
    direction: str,
    ascending: bool,
) -> None:
    ordered = summary.sort_values("diabetes_ratio", ascending=ascending)
    for rank, (_, row) in enumerate(ordered.head(5).iterrows(), 1):
        cluster_id = int(row["cluster_id"])
        rows.append(
            {
                "feature": feature,
                "direction": direction,
                "rank": rank,
                "cluster_id": cluster_id,
                "n_images": int(row["n_images"]),
                "n_mice": int(row["n_mice"]),
                "diabetes_ratio": float(row["diabetes_ratio"]),
                "diabetes_images": int(row["diabetes_images"]),
                "not_diabetes_images": int(row["not_diabetes_images"]),
                "age_distribution": row["age_distribution"],
                "thumbnail": str(
                    output_dir / "clusters" / f"cluster_{cluster_id:02d}" / "thumbnails.html"
                ),
            }
        )


if __name__ == "__main__":
    main()
