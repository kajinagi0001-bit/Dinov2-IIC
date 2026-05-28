from __future__ import annotations

import html
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def run_kmeans(
    metadata_csv: str | Path,
    features_npz: str | Path,
    output_dir: str | Path,
    n_clusters: int,
    seed: int = 42,
    copy_images: bool = True,
    representative_count: int = 30,
) -> None:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Clustering requires scikit-learn. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    metadata = pd.read_csv(metadata_csv)
    data = np.load(features_npz, allow_pickle=True)
    features = data["features"]
    if len(metadata) != features.shape[0]:
        raise ValueError(
            f"Metadata rows ({len(metadata)}) and features ({features.shape[0]}) differ."
        )

    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    cluster_id = model.fit_predict(scaled)
    distances = model.transform(scaled)
    min_distance = distances[np.arange(len(cluster_id)), cluster_id]
    confidence = 1.0 / (1.0 + min_distance)

    assignments = metadata.copy()
    assignments["cluster_id"] = cluster_id
    assignments["cluster_probability"] = confidence
    assignments["entropy"] = np.nan
    assignments["distance_to_center"] = min_distance

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "assignments.csv"
    assignments.to_csv(assignments_path, index=False, encoding="utf-8-sig")

    metrics = _cluster_metrics(scaled, cluster_id, silhouette_score, davies_bouldin_score)
    (output_dir / "metrics.csv").write_text(
        pd.DataFrame([metrics]).to_csv(index=False),
        encoding="utf-8",
    )

    _write_cluster_summary(assignments, output_dir / "cluster_summary.csv")
    _write_report(assignments, metrics, output_dir / "analysis_report.md")
    if copy_images:
        export_cluster_images(assignments, output_dir / "clusters", representative_count)


def export_cluster_images(
    assignments: pd.DataFrame,
    output_root: str | Path,
    representative_count: int = 30,
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for cluster_id, group in assignments.groupby("cluster_id"):
        cluster_dir = output_root / f"cluster_{int(cluster_id):02d}"
        images_dir = cluster_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        ordered = group.sort_values("distance_to_center", kind="stable")
        exported_rows = []
        for _, row in ordered.head(representative_count).iterrows():
            source = Path(row["image_path"])
            if not source.exists():
                continue
            dest = images_dir / f"mouse{int(row['mouse_id']):03d}_{source.name}"
            shutil.copy2(source, dest)
            exported_rows.append((row, dest))
        _write_thumbnail_html(exported_rows, cluster_dir / "thumbnails.html")


def _cluster_metrics(features, cluster_id, silhouette_score, davies_bouldin_score) -> dict:
    metrics = {
        "n_images": int(features.shape[0]),
        "n_clusters": int(len(set(cluster_id))),
    }
    if len(set(cluster_id)) > 1 and features.shape[0] > len(set(cluster_id)):
        metrics["silhouette_score"] = float(silhouette_score(features, cluster_id))
        metrics["davies_bouldin_index"] = float(davies_bouldin_score(features, cluster_id))
    else:
        metrics["silhouette_score"] = np.nan
        metrics["davies_bouldin_index"] = np.nan
    return metrics


def _write_cluster_summary(assignments: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for cluster_id, group in assignments.groupby("cluster_id"):
        label_counts = group["diabetes_label"].value_counts().to_dict()
        age_counts = group["age_week"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_images": len(group),
                "n_mice": group["mouse_id"].nunique(),
                "diabetes_images": label_counts.get("diabetes", 0),
                "not_diabetes_images": label_counts.get("not-diabetes", 0),
                "diabetes_ratio": label_counts.get("diabetes", 0) / len(group),
                "age_distribution": age_counts,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def _write_report(assignments: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    lines = [
        "# Clustering Analysis Report",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Cluster Summary", ""])
    for cluster_id, group in assignments.groupby("cluster_id"):
        labels = group["diabetes_label"].value_counts().to_dict()
        ages = group["age_week"].value_counts().sort_index().to_dict()
        lines.extend(
            [
                f"### Cluster {int(cluster_id):02d}",
                "",
                f"- images: {len(group)}",
                f"- mice: {group['mouse_id'].nunique()}",
                f"- labels: {labels}",
                f"- ages: {ages}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_thumbnail_html(rows: list[tuple[pd.Series, Path]], output_path: Path) -> None:
    cards = []
    for row, image_path in rows:
        label = html.escape(
            f"mouse={row['mouse_id']} age={row['age_week']} label={row['diabetes_label']}"
        )
        rel = html.escape(image_path.relative_to(output_path.parent).as_posix())
        cards.append(
            f"<figure><img src='{rel}' loading='lazy'><figcaption>{label}</figcaption></figure>"
        )
    doc = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<style>
body { font-family: system-ui, sans-serif; margin: 24px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
figure { margin: 0; border: 1px solid #ddd; padding: 8px; }
img { width: 100%; height: 150px; object-fit: contain; background: #f6f6f6; }
figcaption { font-size: 12px; margin-top: 6px; overflow-wrap: anywhere; }
</style>
</head>
<body><main class="grid">
""" + "\n".join(cards) + """
</main></body></html>
"""
    output_path.write_text(doc, encoding="utf-8")
