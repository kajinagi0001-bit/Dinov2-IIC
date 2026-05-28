from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetDirs:
    root: Path
    positive_dir: str
    negative_dir: str


def build_metadata(config: dict, project_dir: str | Path = ".") -> pd.DataFrame:
    project_dir = Path(project_dir)
    dataset_cfg = config["dataset"]
    metadata_cfg = config["metadata"]

    dataset_root = _resolve(project_dir, dataset_cfg["root"])
    metadata_excel = _resolve(project_dir, dataset_cfg["metadata_excel"])
    dirs = DatasetDirs(
        root=dataset_root,
        positive_dir=dataset_cfg.get("positive_dir", "diabetes"),
        negative_dir=dataset_cfg.get("negative_dir", "not-diabetes"),
    )

    image_rows = scan_image_rows(dirs)
    sample_info = load_sample_info(metadata_excel, metadata_cfg)
    rows = pd.DataFrame(image_rows)
    if rows.empty:
        return _empty_metadata()

    merged = rows.merge(sample_info, on="mouse_id", how="left", validate="many_to_one")
    missing = merged[merged["age_week"].isna() | merged["kidney_number"].isna()]
    if not missing.empty:
        examples = missing["image_path"].head(5).tolist()
        raise ValueError(
            "Some images could not be joined to the Excel metadata by mouse_id. "
            f"Examples: {examples}"
        )

    label_mismatch = merged[merged["diabetes_label"] != merged["excel_label"]]
    if not label_mismatch.empty:
        examples = label_mismatch[["image_path", "diabetes_label", "excel_label"]].head(5)
        raise ValueError(
            "Dataset folder labels do not match Excel labels. "
            f"Examples: {examples.to_dict(orient='records')}"
        )

    merged = merged.drop(columns=["excel_label"])
    return merged[
        [
            "image_path",
            "mouse_id",
            "section_id",
            "diabetes_label",
            "age_week",
            "kidney_number",
        ]
    ].sort_values(["mouse_id", "section_id", "image_path"], kind="stable")


def scan_image_rows(dirs: DatasetDirs) -> list[dict]:
    if not dirs.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dirs.root}")

    label_dirs = {
        dirs.positive_dir: "diabetes",
        dirs.negative_dir: "not-diabetes",
    }
    rows: list[dict] = []
    for folder_name, label in label_dirs.items():
        label_root = dirs.root / folder_name
        if not label_root.exists():
            raise FileNotFoundError(f"Required label folder does not exist: {label_root}")
        for image_path in iter_images(label_root):
            section_id = image_path.parent.name
            mouse_id = extract_mouse_id(section_id)
            if mouse_id is None:
                mouse_id = extract_mouse_id(image_path.name)
            if mouse_id is None:
                raise ValueError(
                    "Could not extract leading mouse number from parent folder or file name: "
                    f"{image_path}"
                )
            rows.append(
                {
                    "image_path": str(image_path),
                    "mouse_id": int(mouse_id),
                    "section_id": section_id,
                    "diabetes_label": label,
                }
            )
    return rows


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def extract_mouse_id(name: str) -> int | None:
    match = re.match(r"^\D*(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def load_sample_info(excel_path: Path, metadata_cfg: dict) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Metadata Excel file does not exist: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    mouse_col = metadata_cfg["mouse_id_column"]
    label_col = metadata_cfg["label_column"]
    age_col = metadata_cfg["age_column"]
    kidney_col = metadata_cfg["kidney_number_column"]
    label_mapping = metadata_cfg["label_mapping"]

    missing_cols = [c for c in [mouse_col, label_col, age_col, kidney_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required Excel columns: {missing_cols}")

    out = pd.DataFrame(
        {
            "mouse_id": df[mouse_col].astype(int),
            "excel_label": df[label_col].map(label_mapping),
            "age_week": df[age_col].astype(int),
            "kidney_number": df[kidney_col].astype(int),
        }
    )
    if out["excel_label"].isna().any():
        unknown = df.loc[out["excel_label"].isna(), label_col].unique().tolist()
        raise ValueError(f"Unknown labels in Excel metadata: {unknown}")
    return out


def save_metadata(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def _empty_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "image_path",
            "mouse_id",
            "section_id",
            "diabetes_label",
            "age_week",
            "kidney_number",
        ]
    )


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path
