from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def inspect_metadata(
    metadata_csv: str | Path,
    output_dir: str | Path,
    top_mice: int = 30,
) -> None:
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    _validate_columns(df, metadata_csv)

    overview = _overview(df, metadata_csv)
    label_counts = _count_table(df, "diabetes_label", "label")
    age_counts = _count_table(df, "age_week", "age_week")
    label_age_counts = pd.crosstab(df["age_week"], df["diabetes_label"], margins=True)
    kidney_counts = _count_table(df, "kidney_number", "kidney_number")
    section_counts = _section_counts(df)
    mouse_counts = _mouse_counts(df)
    top_mouse_counts = mouse_counts.head(top_mice)

    _write_csvs(
        output_dir=output_dir,
        overview=overview,
        label_counts=label_counts,
        age_counts=age_counts,
        label_age_counts=label_age_counts,
        kidney_counts=kidney_counts,
        section_counts=section_counts,
        mouse_counts=mouse_counts,
        top_mouse_counts=top_mouse_counts,
    )
    _write_markdown(
        output_dir=output_dir,
        overview=overview,
        label_counts=label_counts,
        age_counts=age_counts,
        label_age_counts=label_age_counts,
        kidney_counts=kidney_counts,
        section_counts=section_counts,
        top_mouse_counts=top_mouse_counts,
    )
    _write_html(
        output_dir=output_dir,
        overview=overview,
        label_counts=label_counts,
        age_counts=age_counts,
        label_age_counts=label_age_counts,
        kidney_counts=kidney_counts,
        section_counts=section_counts,
        top_mouse_counts=top_mouse_counts,
    )


def _validate_columns(df: pd.DataFrame, metadata_csv: Path) -> None:
    required = {
        "image_path",
        "mouse_id",
        "section_id",
        "diabetes_label",
        "age_week",
        "kidney_number",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns in {metadata_csv}: {missing}")


def _overview(df: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    rows = [
        ("metadata_csv", str(metadata_csv)),
        ("images", len(df)),
        ("mice", df["mouse_id"].nunique()),
        ("sections", df["section_id"].nunique()),
        ("labels", df["diabetes_label"].nunique()),
        ("ages", df["age_week"].nunique()),
        ("kidney_numbers", df["kidney_number"].nunique()),
        ("missing_values", int(df.isna().sum().sum())),
        ("min_images_per_mouse", int(df.groupby("mouse_id").size().min())),
        ("median_images_per_mouse", float(df.groupby("mouse_id").size().median())),
        ("max_images_per_mouse", int(df.groupby("mouse_id").size().max())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _count_table(df: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    counts = df[column].value_counts().sort_index()
    out = counts.rename("images").reset_index()
    out.columns = [name, "images"]
    out["image_fraction"] = out["images"] / len(df)
    return out


def _section_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["diabetes_label", "age_week"])
        .agg(images=("image_path", "size"), mice=("mouse_id", "nunique"), sections=("section_id", "nunique"))
        .reset_index()
        .sort_values(["diabetes_label", "age_week"], kind="stable")
    )
    return out


def _mouse_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["mouse_id", "diabetes_label", "age_week", "kidney_number"])
        .agg(images=("image_path", "size"), sections=("section_id", "nunique"))
        .reset_index()
        .sort_values(["images", "mouse_id"], ascending=[False, True], kind="stable")
    )
    out["image_fraction"] = out["images"] / len(df)
    return out


def _write_csvs(
    output_dir: Path,
    overview: pd.DataFrame,
    label_counts: pd.DataFrame,
    age_counts: pd.DataFrame,
    label_age_counts: pd.DataFrame,
    kidney_counts: pd.DataFrame,
    section_counts: pd.DataFrame,
    mouse_counts: pd.DataFrame,
    top_mouse_counts: pd.DataFrame,
) -> None:
    overview.to_csv(output_dir / "overview.csv", index=False, encoding="utf-8-sig")
    label_counts.to_csv(output_dir / "label_counts.csv", index=False, encoding="utf-8-sig")
    age_counts.to_csv(output_dir / "age_counts.csv", index=False, encoding="utf-8-sig")
    label_age_counts.to_csv(output_dir / "label_age_counts.csv", encoding="utf-8-sig")
    kidney_counts.to_csv(output_dir / "kidney_counts.csv", index=False, encoding="utf-8-sig")
    section_counts.to_csv(output_dir / "label_age_section_counts.csv", index=False, encoding="utf-8-sig")
    mouse_counts.to_csv(output_dir / "mouse_image_counts.csv", index=False, encoding="utf-8-sig")
    top_mouse_counts.to_csv(output_dir / "top_mouse_image_counts.csv", index=False, encoding="utf-8-sig")


def _write_markdown(
    output_dir: Path,
    overview: pd.DataFrame,
    label_counts: pd.DataFrame,
    age_counts: pd.DataFrame,
    label_age_counts: pd.DataFrame,
    kidney_counts: pd.DataFrame,
    section_counts: pd.DataFrame,
    top_mouse_counts: pd.DataFrame,
) -> None:
    lines = [
        "# Metadata Inspection",
        "",
        "## Overview",
        "",
        _to_markdown(overview),
        "",
        "## Label Counts",
        "",
        _to_markdown(label_counts),
        "",
        "## Age Counts",
        "",
        _to_markdown(age_counts),
        "",
        "## Age x Label Counts",
        "",
        _to_markdown(label_age_counts.reset_index()),
        "",
        "## Label x Age Section Counts",
        "",
        _to_markdown(section_counts),
        "",
        "## Top Kidney Numbers",
        "",
        _to_markdown(kidney_counts.head(20)),
        "",
        "## Top Mice By Image Count",
        "",
        _to_markdown(top_mouse_counts),
        "",
        "HTML charts: `inspection.html`",
        "",
    ]
    (output_dir / "inspection_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_html(
    output_dir: Path,
    overview: pd.DataFrame,
    label_counts: pd.DataFrame,
    age_counts: pd.DataFrame,
    label_age_counts: pd.DataFrame,
    kidney_counts: pd.DataFrame,
    section_counts: pd.DataFrame,
    top_mouse_counts: pd.DataFrame,
) -> None:
    cards = "\n".join(
        f"<article><strong>{_esc(row.metric)}</strong><span>{_esc(row.value)}</span></article>"
        for row in overview.itertuples(index=False)
    )
    doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Metadata Inspection</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 28px; color: #202124; }}
h1, h2 {{ margin: 0 0 14px; }}
section {{ margin: 0 0 34px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
article {{ border: 1px solid #d8dce0; padding: 10px; border-radius: 6px; background: #fbfbfc; }}
article strong {{ display: block; font-size: 12px; color: #5f6368; }}
article span {{ display: block; margin-top: 6px; font-size: 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
svg {{ width: 100%; max-width: 980px; height: auto; display: block; }}
.note {{ color: #5f6368; font-size: 13px; }}
</style>
</head>
<body>
<h1>Metadata Inspection</h1>
<section><h2>Overview</h2><div class="cards">{cards}</div></section>
<section><h2>Label Distribution</h2>{_bar_chart(label_counts, "label", "images")}{_html_table(label_counts)}</section>
<section><h2>Age Distribution</h2>{_bar_chart(age_counts, "age_week", "images")}{_html_table(age_counts)}</section>
<section><h2>Age x Label</h2>{_stacked_age_label_chart(label_age_counts)}{_html_table(label_age_counts.reset_index())}</section>
<section><h2>Top Kidney Numbers</h2>{_bar_chart(kidney_counts.head(20), "kidney_number", "images")}{_html_table(kidney_counts.head(20))}</section>
<section><h2>Label x Age Summary</h2>{_html_table(section_counts)}</section>
<section><h2>Top Mice By Image Count</h2>{_bar_chart(top_mouse_counts, "mouse_id", "images")}{_html_table(top_mouse_counts)}</section>
<p class="note">Generated from metadata only. Labels and ages are for inspection, not training.</p>
</body>
</html>
"""
    (output_dir / "inspection.html").write_text(doc, encoding="utf-8")


def _bar_chart(df: pd.DataFrame, label_col: str, value_col: str) -> str:
    if df.empty:
        return "<p>No data.</p>"
    width = 980
    row_h = 28
    left = 170
    right = 80
    height = 30 + row_h * len(df)
    max_value = max(float(df[value_col].max()), 1.0)
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img'>"]
    for index, row in enumerate(df.itertuples(index=False), 0):
        y = 22 + index * row_h
        label = str(getattr(row, label_col))
        value = float(getattr(row, value_col))
        bar_w = (width - left - right) * value / max_value
        parts.append(f"<text x='0' y='{y + 16}' font-size='13'>{_esc(label)}</text>")
        parts.append(f"<rect x='{left}' y='{y}' width='{bar_w:.1f}' height='18' fill='#3b82f6'></rect>")
        parts.append(f"<text x='{left + bar_w + 6:.1f}' y='{y + 14}' font-size='12'>{int(value):,}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def _stacked_age_label_chart(table: pd.DataFrame) -> str:
    data = table.drop(index="All", errors="ignore").drop(columns="All", errors="ignore")
    if data.empty:
        return "<p>No data.</p>"
    width = 980
    row_h = 32
    left = 90
    right = 120
    height = 34 + row_h * len(data)
    colors = ["#3b82f6", "#f97316", "#10b981", "#8b5cf6"]
    max_total = max(float(data.sum(axis=1).max()), 1.0)
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img'>"]
    columns = list(data.columns)
    for index, (age, row) in enumerate(data.iterrows()):
        y = 24 + index * row_h
        x = left
        total = float(row.sum())
        parts.append(f"<text x='0' y='{y + 16}' font-size='13'>{_esc(age)}</text>")
        for c_index, column in enumerate(columns):
            value = float(row[column])
            bar_w = (width - left - right) * value / max_total
            parts.append(
                f"<rect x='{x:.1f}' y='{y}' width='{bar_w:.1f}' height='20' fill='{colors[c_index % len(colors)]}'></rect>"
            )
            x += bar_w
        parts.append(f"<text x='{x + 6:.1f}' y='{y + 15}' font-size='12'>{int(total):,}</text>")
    legend_x = left
    for c_index, column in enumerate(columns):
        parts.append(
            f"<rect x='{legend_x}' y='0' width='12' height='12' fill='{colors[c_index % len(colors)]}'></rect>"
        )
        parts.append(f"<text x='{legend_x + 16}' y='11' font-size='12'>{_esc(column)}</text>")
        legend_x += 150
    parts.append("</svg>")
    return "\n".join(parts)


def _html_table(df: pd.DataFrame) -> str:
    headers = "".join(f"<th>{_esc(col)}</th>" for col in df.columns)
    rows = []
    for row in df.itertuples(index=False):
        rows.append("".join(f"<td>{_esc(value)}</td>" for value in row))
    body = "\n".join(f"<tr>{row}</tr>" for row in rows)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_md(value) for value in row) + " |")
    return "\n".join(lines)


def _md(value) -> str:
    return str(value).replace("|", "\\|")


def _esc(value) -> str:
    return html.escape(str(value), quote=True)
