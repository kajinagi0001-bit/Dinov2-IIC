from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def create_umap_visualization(
    features_npz: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    assignments_csv: str | Path | None = None,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "cosine",
    seed: int = 42,
    width: int = 1800,
    height: int = 1400,
    point_radius: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_npz = Path(features_npz)
    metadata = pd.read_csv(metadata_csv)
    features_data = np.load(features_npz, allow_pickle=True)
    features = features_data["features"].astype(np.float32)
    if len(metadata) != features.shape[0]:
        raise ValueError(
            f"metadata rows ({len(metadata)}) and features ({features.shape[0]}) differ"
        )

    reducer = _make_umap(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, seed=seed)
    coords = reducer.fit_transform(features)

    out = metadata.copy()
    out["umap_x"] = coords[:, 0]
    out["umap_y"] = coords[:, 1]
    if assignments_csv is not None:
        assignments = pd.read_csv(assignments_csv)
        if len(assignments) != len(out):
            raise ValueError(
                f"assignments rows ({len(assignments)}) and metadata rows ({len(out)}) differ"
            )
        out["cluster_id"] = assignments["cluster_id"].astype(int).to_numpy()
        if "cluster_probability" in assignments.columns:
            out["cluster_probability"] = assignments["cluster_probability"].to_numpy()
        if "entropy" in assignments.columns:
            out["entropy"] = assignments["entropy"].to_numpy()

    coords_path = output_dir / "umap_coordinates.csv"
    out.to_csv(coords_path, index=False, encoding="utf-8-sig")

    plot_paths = []
    plot_paths.append(
        _scatter_png(
            out,
            color_col="diabetes_label",
            output_path=output_dir / "umap_by_label.png",
            width=width,
            height=height,
            point_radius=point_radius,
            title="UMAP by diabetes label",
        )
    )
    plot_paths.append(
        _scatter_png(
            out,
            color_col="age_week",
            output_path=output_dir / "umap_by_age.png",
            width=width,
            height=height,
            point_radius=point_radius,
            title="UMAP by age week",
        )
    )
    if "cluster_id" in out.columns:
        plot_paths.append(
            _scatter_png(
                out,
                color_col="cluster_id",
                output_path=output_dir / "umap_by_iic_cluster.png",
                width=width,
                height=height,
                point_radius=point_radius,
                title="UMAP by IIC cluster",
            )
        )

    _write_html(
        output_dir,
        coords_path,
        plot_paths,
        features_npz,
        n_neighbors,
        min_dist,
        metric,
        seed,
        point_radius,
    )
    _write_summary(
        out,
        output_dir / "umap_summary.md",
        features_npz,
        n_neighbors,
        min_dist,
        metric,
        seed,
        point_radius,
    )


def _make_umap(n_neighbors: int, min_dist: float, metric: str, seed: int):
    try:
        from umap import UMAP
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "UMAP visualization requires umap-learn. Install with `pip install umap-learn`."
        ) from exc
    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        low_memory=True,
        verbose=True,
    )


def _scatter_png(
    df: pd.DataFrame,
    color_col: str,
    output_path: Path,
    width: int,
    height: int,
    point_radius: int,
    title: str,
) -> Path:
    pad_left, pad_right, pad_top, pad_bottom = 90, 280, 70, 80
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    x = df["umap_x"].to_numpy(dtype=float)
    y = df["umap_y"].to_numpy(dtype=float)
    px = pad_left + _scale(x, plot_w)
    py = pad_top + plot_h - _scale(y, plot_h)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    draw.text((pad_left, 22), title, fill=(20, 24, 28), font=font)
    draw.rectangle(
        (pad_left, pad_top, pad_left + plot_w, pad_top + plot_h),
        outline=(210, 214, 220, 255),
        width=1,
    )

    values = df[color_col].astype(str).fillna("NA").to_numpy()
    unique = sorted(pd.unique(values), key=_sort_key)
    palette = _palette(len(unique))
    color_map = {value: palette[index] for index, value in enumerate(unique)}

    for value in unique:
        mask = values == value
        color = color_map[value]
        for x_i, y_i in zip(px[mask], py[mask]):
            draw.ellipse(
                (
                    x_i - point_radius,
                    y_i - point_radius,
                    x_i + point_radius,
                    y_i + point_radius,
                ),
                fill=color,
            )

    legend_x = pad_left + plot_w + 24
    legend_y = pad_top
    draw.text((legend_x, legend_y - 26), color_col, fill=(20, 24, 28), font=font)
    for index, value in enumerate(unique[:40]):
        y0 = legend_y + index * 24
        color = color_map[value]
        draw.rectangle((legend_x, y0, legend_x + 14, y0 + 14), fill=color)
        draw.text((legend_x + 22, y0), str(value), fill=(20, 24, 28), font=font)
    if len(unique) > 40:
        draw.text((legend_x, legend_y + 40 * 24 + 8), f"... {len(unique) - 40} more", fill=(80, 86, 94), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return output_path


def _scale(values: np.ndarray, size: int) -> np.ndarray:
    vmin = np.nanpercentile(values, 0.5)
    vmax = np.nanpercentile(values, 99.5)
    if vmax <= vmin:
        vmax = float(np.nanmax(values))
        vmin = float(np.nanmin(values))
    if vmax <= vmin:
        return np.full_like(values, size / 2)
    clipped = np.clip(values, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin) * size


def _palette(n: int) -> list[tuple[int, int, int, int]]:
    base = [
        (37, 99, 235, 155),
        (220, 38, 38, 155),
        (22, 163, 74, 155),
        (234, 88, 12, 155),
        (124, 58, 237, 155),
        (8, 145, 178, 155),
        (202, 138, 4, 155),
        (219, 39, 119, 155),
        (75, 85, 99, 155),
        (13, 148, 136, 155),
    ]
    if n <= len(base):
        return base[:n]
    colors = []
    for i in range(n):
        hue = i / max(1, n)
        colors.append(_hsv_to_rgba(hue, 0.72, 0.82, 155))
    return colors


def _hsv_to_rgba(h: float, s: float, v: float, a: int) -> tuple[int, int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255), a


def _sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _write_html(
    output_dir: Path,
    coords_path: Path,
    plot_paths: list[Path],
    features_npz: Path,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
    point_radius: int,
) -> None:
    images = "\n".join(
        f"<section><h2>{html.escape(path.stem)}</h2><img src='{html.escape(path.name)}'></section>"
        for path in plot_paths
    )
    doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>UMAP Feature Map</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 28px; color: #202124; }}
img {{ max-width: 100%; border: 1px solid #d8dce0; }}
code {{ background: #f1f3f4; padding: 2px 4px; border-radius: 4px; }}
section {{ margin-bottom: 32px; }}
</style>
</head>
<body>
<h1>UMAP Feature Map</h1>
<p>features: <code>{html.escape(str(features_npz))}</code></p>
<p>coordinates: <code>{html.escape(coords_path.name)}</code></p>
<p>n_neighbors={n_neighbors}, min_dist={min_dist}, metric={html.escape(metric)}, seed={seed}, point_radius={point_radius}</p>
{images}
</body>
</html>
"""
    (output_dir / "umap.html").write_text(doc, encoding="utf-8")


def _write_summary(
    df: pd.DataFrame,
    output_path: Path,
    features_npz: Path,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
    point_radius: int,
) -> None:
    lines = [
        "# UMAP Feature Map",
        "",
        f"- features: `{features_npz}`",
        f"- images: {len(df)}",
        f"- n_neighbors: {n_neighbors}",
        f"- min_dist: {min_dist}",
        f"- metric: `{metric}`",
        f"- seed: {seed}",
        f"- point_radius: {point_radius}",
        "",
        "## Outputs",
        "",
        "- `umap_coordinates.csv`",
        "- `umap_by_label.png`",
        "- `umap_by_age.png`",
    ]
    if "cluster_id" in df.columns:
        lines.append("- `umap_by_iic_cluster.png`")
    lines.append("- `umap.html`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
