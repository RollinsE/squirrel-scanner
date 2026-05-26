"""Utilities for merging multiple YOLO detection datasets.

The main use case is combining several Roboflow YOLO exports whose class lists do
not match exactly. Labels are remapped into one canonical class list and copied
into a fresh YOLO dataset root with a new data.yaml.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from src.logger import component_stage, log_stage

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("valid", "val"),
    "test": ("test",),
}

DEFAULT_CLASS_ALIASES: Mapping[str, str] = {
    "red": "red",
    "red squirrel": "red",
    "red squirrels": "red",
    "red_squirrel": "red",
    "red_squirrels": "red",
    "redsquirrel": "red",
    "redsquirrels": "red",
    "grey": "grey",
    "gray": "grey",
    "grey squirrel": "grey",
    "grey squirrels": "grey",
    "gray squirrel": "grey",
    "gray squirrels": "grey",
    "grey_squirrel": "grey",
    "grey_squirrels": "grey",
    "gray_squirrel": "grey",
    "gray_squirrels": "grey",
    "greysquirrel": "grey",
    "greysquirrels": "grey",
    "marten": "marten",
    "martens": "marten",
    "pine marten": "marten",
    "pine martens": "marten",
    "pine_marten": "marten",
    "pine_martens": "marten",
    "rat": "rat",
    "rats": "rat",
}


def _normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _safe_slug(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "dataset"


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def parse_class_map(value: Optional[str]) -> Dict[str, str]:
    """Parse class aliases from CLI text.

    Accepted separators:
      red squirrels=red,grey squirrels=grey
      red squirrels:red;grey squirrels:grey
    """
    if not value:
        return {}

    mapping: Dict[str, str] = {}
    chunks = [c.strip() for c in re.split(r"[;,]", value) if c.strip()]
    for chunk in chunks:
        if "=" in chunk:
            src, dst = chunk.split("=", 1)
        elif ":" in chunk:
            src, dst = chunk.split(":", 1)
        else:
            raise ValueError(
                "Invalid --class_map entry. Use 'source=target' pairs, e.g. "
                "'red squirrels=red,grey squirrels=grey,rats=rat'."
            )
        mapping[_normalize_name(src)] = _safe_slug(dst)
    return mapping


def canonical_class_name(name: Any, class_map: Optional[Mapping[str, str]] = None) -> str:
    norm = _normalize_name(name)
    merged: Dict[str, str] = {k: v for k, v in DEFAULT_CLASS_ALIASES.items()}
    if class_map:
        merged.update({_normalize_name(k): _safe_slug(v) for k, v in class_map.items()})
    if norm in merged:
        return merged[norm]
    compact = norm.replace(" ", "")
    if compact in merged:
        return merged[compact]
    return _safe_slug(norm)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected data.yaml at: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return data


def _extract_names(data: Mapping[str, Any], yaml_path: Path) -> List[str]:
    names = data.get("names")
    if isinstance(names, dict):
        ordered = [names[k] for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
        return [str(x) for x in ordered]
    if isinstance(names, list):
        return [str(x) for x in names]
    raise ValueError(f"data.yaml has no usable 'names' list/dict: {yaml_path}")


def _find_existing_split_dir(root: Path, split: str, kind: str) -> Optional[Path]:
    for alias in SPLIT_ALIASES.get(split, (split,)):
        candidate = root / alias / kind
        if candidate.is_dir():
            return candidate
    return None


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _copy_or_empty_label(
    *,
    src_label: Path,
    dst_label: Path,
    source_to_target_id: Mapping[int, int],
    report: Dict[str, Any],
) -> None:
    remapped: List[str] = []
    if src_label.is_file():
        for raw_line in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                source_id = int(float(parts[0]))
            except Exception:
                report["bad_label_lines"] += 1
                continue
            target_id = source_to_target_id.get(source_id)
            if target_id is None:
                report["dropped_label_lines"] += 1
                continue
            parts[0] = str(target_id)
            remapped.append(" ".join(parts))
    else:
        report["missing_label_files"] += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
    report["written_label_lines"] += len(remapped)


def _ensure_dataset_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {path}")
    if not (root / "data.yaml").is_file():
        raise FileNotFoundError(f"Expected data.yaml at: {root / 'data.yaml'}")
    return root


def merge_yolo_detection_datasets(
    *,
    dataset_roots: Sequence[str],
    output_dir: str,
    target_classes: Optional[Sequence[str]] = None,
    class_map: Optional[Mapping[str, str]] = None,
    run_dir: Optional[str] = None,
    LOG=None,
) -> str:
    """Merge YOLO detection datasets into one class-consistent dataset.

    Returns the merged dataset root. The output dataset uses train/valid/test
    folders and a data.yaml that points to those folders.
    """
    if len(dataset_roots) < 1:
        raise ValueError("At least one dataset root is required.")

    roots = [_ensure_dataset_root(p) for p in dataset_roots]
    out_root = Path(output_dir).expanduser().resolve()

    for root in roots:
        if out_root == root:
            raise ValueError(f"Merged dataset output cannot overwrite a source dataset: {out_root}")
        if out_root in root.parents:
            raise ValueError(
                f"Merged dataset output cannot be a parent of a source dataset: {out_root}. "
                "Choose a dedicated folder such as data/merged_squirrel_rat."
            )
        if root in out_root.parents:
            raise ValueError(
                f"Merged dataset output cannot be inside a source dataset: {out_root}. "
                "Choose a sibling folder such as data/merged_squirrel_rat."
            )

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    user_map = dict(class_map or {})
    target_class_list = [canonical_class_name(c, user_map) for c in (target_classes or [])]
    if target_class_list:
        # Preserve order while removing duplicates.
        seen = set()
        target_class_list = [c for c in target_class_list if not (c in seen or seen.add(c))]
    else:
        discovered: List[str] = []
        for root in roots:
            data = _read_yaml(root / "data.yaml")
            for name in _extract_names(data, root / "data.yaml"):
                canonical = canonical_class_name(name, user_map)
                if canonical not in discovered:
                    discovered.append(canonical)
        target_class_list = discovered

    if not target_class_list:
        raise ValueError("Could not determine any target classes to write into merged data.yaml.")

    target_id = {name: idx for idx, name in enumerate(target_class_list)}

    unknown_classes: List[Dict[str, str]] = []
    for root in roots:
        yaml_path = root / "data.yaml"
        data = _read_yaml(yaml_path)
        for source_name in _extract_names(data, yaml_path):
            canonical = canonical_class_name(source_name, user_map)
            if canonical not in target_id:
                unknown_classes.append({
                    "dataset": str(root),
                    "source_class": str(source_name),
                    "resolved_class": canonical,
                })

    if unknown_classes:
        examples = "; ".join(
            f"{item['source_class']} -> {item['resolved_class']} in {item['dataset']}"
            for item in unknown_classes[:8]
        )
        allowed = ",".join(target_class_list)
        raise ValueError(
            "One or more source classes do not match --target_classes after applying --class_map. "
            f"Allowed target classes: {allowed}. Unmatched examples: {examples}. "
            "Add the missing class to --target_classes or map it with --class_map, "
            "for example 'source name=target_name'."
        )

    report: Dict[str, Any] = {
        "version": 1,
        "output_dir": str(out_root),
        "target_classes": target_class_list,
        "source_datasets": [],
        "splits": {
            "train": {"images": 0, "labels": 0},
            "valid": {"images": 0, "labels": 0},
            "test": {"images": 0, "labels": 0},
        },
        "missing_label_files": 0,
        "bad_label_lines": 0,
        "dropped_label_lines": 0,
        "written_label_lines": 0,
    }

    ctx = (
        component_stage(
            LOG,
            "DatasetMerge",
            dataset_count=len(roots),
            output_dir=str(out_root),
            target_classes=target_class_list,
        )
        if LOG is not None
        else nullcontext()
    )

    with ctx:
        for canonical_split in ("train", "valid", "test"):
            (out_root / canonical_split / "images").mkdir(parents=True, exist_ok=True)
            (out_root / canonical_split / "labels").mkdir(parents=True, exist_ok=True)

        for ds_idx, root in enumerate(roots):
            yaml_path = root / "data.yaml"
            data = _read_yaml(yaml_path)
            source_names = _extract_names(data, yaml_path)
            source_to_target: Dict[int, int] = {}
            source_map: Dict[str, Optional[str]] = {}
            for i, source_name in enumerate(source_names):
                canonical = canonical_class_name(source_name, user_map)
                source_map[str(source_name)] = canonical if canonical in target_id else None
                if canonical in target_id:
                    source_to_target[i] = target_id[canonical]

            source_report: Dict[str, Any] = {
                "root": str(root),
                "source_names": source_names,
                "source_to_target_class": source_map,
                "splits": {},
            }

            for canonical_split in ("train", "valid", "test"):
                images_dir = _find_existing_split_dir(root, canonical_split, "images")
                labels_dir = _find_existing_split_dir(root, canonical_split, "labels")
                if images_dir is None:
                    source_report["splits"][canonical_split] = {"images": 0, "labels": 0, "status": "missing"}
                    continue

                split_images = 0
                split_labels = 0
                source_slug = _safe_slug(root.name)
                for img_path in _iter_images(images_dir):
                    new_stem = f"d{ds_idx:02d}_{source_slug}_{canonical_split}_{_safe_slug(img_path.stem)}"
                    dst_img = out_root / canonical_split / "images" / f"{new_stem}{img_path.suffix.lower()}"
                    dst_label = out_root / canonical_split / "labels" / f"{new_stem}.txt"

                    shutil.copy2(img_path, dst_img)
                    split_images += 1
                    report["splits"][canonical_split]["images"] += 1

                    src_label = (labels_dir / f"{img_path.stem}.txt") if labels_dir else Path("__missing__")
                    before = report["written_label_lines"]
                    _copy_or_empty_label(
                        src_label=src_label,
                        dst_label=dst_label,
                        source_to_target_id=source_to_target,
                        report=report,
                    )
                    if report["written_label_lines"] > before:
                        split_labels += 1
                        report["splits"][canonical_split]["labels"] += 1

                source_report["splits"][canonical_split] = {
                    "images": split_images,
                    "labels_with_boxes": split_labels,
                    "status": "ok",
                }

            report["source_datasets"].append(source_report)

        data_yaml = {
            "path": str(out_root),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(target_class_list),
            "names": target_class_list,
        }
        with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)

        report_path = out_root / "merge_report.json"
        report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

        if run_dir:
            artifacts = Path(run_dir) / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_path, artifacts / "dataset_merge_report.json")

        if LOG is not None:
            log_stage(
                LOG,
                "RESULT",
                "DatasetMerge",
                output_dir=str(out_root),
                target_classes=target_class_list,
                train_images=report["splits"]["train"]["images"],
                valid_images=report["splits"]["valid"]["images"],
                test_images=report["splits"]["test"]["images"],
                report_path=str(report_path),
            )

        return str(out_root)
