import glob
import json
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image

from src.logger import log_stage, component_stage


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "val", "test")
EPS = 1e-6


def _ensure_dataset_root(raw_dataset: str) -> str:
    if not raw_dataset or not os.path.isdir(raw_dataset):
        raise FileNotFoundError(f"--raw_dataset is not a directory: {raw_dataset}")
    data_yaml = os.path.join(raw_dataset, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Expected data.yaml at: {data_yaml}")
    return raw_dataset


def _load_data_yaml(dataset_root: str) -> Dict[str, Any]:
    data_yaml = os.path.join(dataset_root, "data.yaml")
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", [])
    if isinstance(names, dict):
        try:
            names = [names[k] for k in sorted(names, key=lambda x: int(x))]
        except Exception:
            names = list(names.values())

    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid 'names' in yaml: {data_yaml}")

    data["names"] = names
    return data


def _list_images(images_dir: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return sorted(files)


def _find_image_for_stem(images_dir: str, stem: str) -> Optional[str]:
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
        p2 = os.path.join(images_dir, stem + ext.upper())
        if os.path.exists(p2):
            return p2

    hits = glob.glob(os.path.join(images_dir, f"{stem}.*"))
    return hits[0] if hits else None


def _safe_relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def _make_issue(
    *,
    severity: str,
    split: str,
    issue_type: str,
    path: str,
    dataset_root: str,
    line_no: Optional[int] = None,
    value: Optional[str] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "severity": severity,
        "split": split,
        "type": issue_type,
        "file": _safe_relpath(path, dataset_root),
    }
    if line_no is not None:
        item["line"] = int(line_no)
    if value is not None:
        item["value"] = str(value)
    return item


def _verify_image(img_path: str) -> Optional[str]:
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return f"invalid_image_size_{w}x{h}"
        return None
    except Exception as e:
        return str(e)


def _format_float(v: float) -> str:
    return f"{v:.15g}"


def _format_clean_label_row(cls: int, xc: float, yc: float, bw: float, bh: float) -> str:
    return f"{cls} {_format_float(xc)} {_format_float(yc)} {_format_float(bw)} {_format_float(bh)}"


def _parse_detect_row(parts: List[str]) -> Tuple[int, float, float, float, float]:
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    return cls, xc, yc, bw, bh


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _clean_label_line(
    line: str,
    *,
    num_classes: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns:
      (cleaned_line, reason)

    Cases:
      - unchanged valid line -> (original_stripped_line, None)
      - fixed valid line     -> (rewritten_line, "clamped_box_to_image_bounds")
      - removed bad line     -> (None, "<reason>")
    """
    stripped = line.strip()
    if not stripped:
        return None, None

    parts = stripped.split()

    if len(parts) > 5:
        return None, "removed_segmentation_row_in_detect_dataset"

    if len(parts) < 5:
        return None, "removed_malformed_label_row"

    try:
        cls, xc, yc, bw, bh = _parse_detect_row(parts)
    except Exception:
        return None, "removed_non_numeric_label_values"

    if not _all_finite(xc, yc, bw, bh):
        return None, "removed_non_finite_label_values"

    if cls < 0 or cls >= num_classes:
        return None, "removed_class_out_of_range"

    if bw <= EPS or bh <= EPS:
        return None, "removed_non_positive_box_size"

    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0

    # Entirely outside the image -> remove
    if x2 <= EPS or x1 >= 1.0 - EPS or y2 <= EPS or y1 >= 1.0 - EPS:
        return None, "removed_box_entirely_outside_image"

    cx1 = _clamp01(x1)
    cy1 = _clamp01(y1)
    cx2 = _clamp01(x2)
    cy2 = _clamp01(y2)

    new_bw = cx2 - cx1
    new_bh = cy2 - cy1
    if new_bw <= EPS or new_bh <= EPS:
        return None, "removed_box_degenerate_after_clamp"

    new_xc = (cx1 + cx2) / 2.0
    new_yc = (cy1 + cy2) / 2.0

    changed = (
        abs(new_xc - xc) > EPS
        or abs(new_yc - yc) > EPS
        or abs(new_bw - bw) > EPS
        or abs(new_bh - bh) > EPS
        or xc < -EPS
        or xc > 1.0 + EPS
        or yc < -EPS
        or yc > 1.0 + EPS
        or bw < -EPS
        or bw > 1.0 + EPS
        or bh < -EPS
        or bh > 1.0 + EPS
    )

    if not changed:
        return stripped, None

    return _format_clean_label_row(cls, new_xc, new_yc, new_bw, new_bh), "clamped_box_to_image_bounds"


def _scan_label_line(
    line: str,
    *,
    num_classes: int,
) -> Tuple[Optional[str], Optional[str]]:
    stripped = line.strip()
    if not stripped:
        return None, None

    parts = stripped.split()

    if len(parts) > 5:
        return "segmentation_row_in_detect_dataset", stripped

    if len(parts) < 5:
        return "malformed_label_row", stripped

    try:
        cls, xc, yc, bw, bh = _parse_detect_row(parts)
    except Exception:
        return "non_numeric_label_values", stripped

    if not _all_finite(xc, yc, bw, bh):
        return "non_finite_label_values", stripped

    if cls < 0 or cls >= num_classes:
        return "class_out_of_range", stripped

    if xc < -EPS or xc > 1.0 + EPS or yc < -EPS or yc > 1.0 + EPS or bw < -EPS or bw > 1.0 + EPS or bh < -EPS or bh > 1.0 + EPS:
        return "coords_not_normalized_0_1", stripped

    if bw <= EPS or bh <= EPS:
        return "non_positive_box_size", stripped

    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0

    if x1 < -EPS or x2 > 1.0 + EPS:
        return "box_outside_image_x_bounds", stripped

    if y1 < -EPS or y2 > 1.0 + EPS:
        return "box_outside_image_y_bounds", stripped

    return None, None


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _backup_file_once(src_path: str, backup_root: Optional[str], dataset_root: str) -> Optional[str]:
    if not backup_root:
        return None
    rel = _safe_relpath(src_path, dataset_root)
    dst = os.path.join(backup_root, rel)
    if not os.path.exists(dst):
        _ensure_parent(dst)
        shutil.copy2(src_path, dst)
    return dst


def _scan_split(
    *,
    dataset_root: str,
    split: str,
    num_classes: int,
) -> Dict[str, Any]:
    split_dir = os.path.join(dataset_root, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    out: Dict[str, Any] = {
        "split": split,
        "images": 0,
        "label_files": 0,
        "background_images": 0,
        "empty_label_files": 0,
        "bad_images": 0,
        "bad_label_files": 0,
        "issues": [],
    }

    if not os.path.isdir(split_dir):
        return out

    if not os.path.isdir(images_dir):
        out["issues"].append(
            _make_issue(
                severity="fatal",
                split=split,
                issue_type="missing_images_dir",
                path=images_dir,
                dataset_root=dataset_root,
            )
        )
        return out

    if not os.path.isdir(labels_dir):
        out["issues"].append(
            _make_issue(
                severity="fatal",
                split=split,
                issue_type="missing_labels_dir",
                path=labels_dir,
                dataset_root=dataset_root,
            )
        )
        return out

    image_files = _list_images(images_dir)
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

    out["images"] = len(image_files)
    out["label_files"] = len(label_files)

    for img_path in image_files:
        img_err = _verify_image(img_path)
        if img_err is not None:
            out["bad_images"] += 1
            out["issues"].append(
                _make_issue(
                    severity="fatal",
                    split=split,
                    issue_type="bad_image",
                    path=img_path,
                    dataset_root=dataset_root,
                    value=img_err,
                )
            )

        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")

        # Missing label file is allowed: background / hard-negative image
        if not os.path.exists(label_path):
            out["background_images"] += 1
            continue

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception as e:
            out["bad_label_files"] += 1
            out["issues"].append(
                _make_issue(
                    severity="fatal",
                    split=split,
                    issue_type="unreadable_label_file",
                    path=label_path,
                    dataset_root=dataset_root,
                    value=str(e),
                )
            )
            continue

        # Empty label file is allowed: background / hard-negative image
        if not lines:
            out["empty_label_files"] += 1
            out["background_images"] += 1
            continue

        file_has_error = False
        for i, line in enumerate(lines, start=1):
            issue_type, raw = _scan_label_line(line, num_classes=num_classes)
            if issue_type is not None:
                file_has_error = True
                out["issues"].append(
                    _make_issue(
                        severity="fatal",
                        split=split,
                        issue_type=issue_type,
                        path=label_path,
                        dataset_root=dataset_root,
                        line_no=i,
                        value=raw,
                    )
                )

        if file_has_error:
            out["bad_label_files"] += 1

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = _find_image_for_stem(images_dir, stem)
        if img_path is None:
            out["bad_label_files"] += 1
            out["issues"].append(
                _make_issue(
                    severity="fatal",
                    split=split,
                    issue_type="label_without_matching_image",
                    path=label_path,
                    dataset_root=dataset_root,
                )
            )

    return out


def auto_fix_yolo_detection_dataset(
    raw_dataset: str,
    run_dir: Optional[str],
    LOG,
) -> Dict[str, Any]:
    """
    Auto-fix safe YOLO detection label issues.

    Safe fixes performed:
      - remove segmentation rows from detect labels
      - remove malformed rows
      - remove non-numeric / non-finite rows
      - remove rows with invalid class ids
      - clamp boxes that spill outside image bounds
      - remove boxes that are entirely outside image or degenerate
      - delete orphan label files with no matching image

    Missing label files and empty label files are kept as valid background / hard-negative images.
    """
    dataset_root = _ensure_dataset_root(raw_dataset)
    data = _load_data_yaml(dataset_root)
    num_classes = len(data["names"])

    artifacts_dir = os.path.join(run_dir, "artifacts") if run_dir else None
    backup_root = os.path.join(artifacts_dir, "dataset_label_backups") if artifacts_dir else None
    report_path = os.path.join(artifacts_dir, "dataset_autofix.json") if artifacts_dir else None

    report: Dict[str, Any] = {
        "dataset_root": dataset_root,
        "num_classes": num_classes,
        "splits": {},
        "files_scanned": 0,
        "label_files_modified": 0,
        "rows_removed": 0,
        "rows_clamped": 0,
        "orphan_label_files_deleted": 0,
        "backups_written": 0,
        "changes": [],
    }

    with component_stage(LOG, "DatasetAutoFix", dataset_root=dataset_root, run_dir=run_dir):
        for split in SPLITS:
            split_dir = os.path.join(dataset_root, split)
            images_dir = os.path.join(split_dir, "images")
            labels_dir = os.path.join(split_dir, "labels")

            split_summary = {
                "label_files_scanned": 0,
                "label_files_modified": 0,
                "rows_removed": 0,
                "rows_clamped": 0,
                "orphan_label_files_deleted": 0,
            }

            if not os.path.isdir(split_dir) or not os.path.isdir(labels_dir):
                report["splits"][split] = split_summary
                continue

            label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
            report["files_scanned"] += len(label_files)
            split_summary["label_files_scanned"] = len(label_files)

            for label_path in label_files:
                stem = os.path.splitext(os.path.basename(label_path))[0]
                matching_image = _find_image_for_stem(images_dir, stem)

                if matching_image is None:
                    backup_path = _backup_file_once(label_path, backup_root, dataset_root)
                    if backup_path is not None:
                        report["backups_written"] += 1

                    os.remove(label_path)
                    report["orphan_label_files_deleted"] += 1
                    split_summary["orphan_label_files_deleted"] += 1
                    report["changes"].append(
                        {
                            "split": split,
                            "file": _safe_relpath(label_path, dataset_root),
                            "action": "deleted_orphan_label_file",
                        }
                    )
                    continue

                try:
                    with open(label_path, "r", encoding="utf-8") as f:
                        original_lines = f.readlines()
                except Exception:
                    continue

                new_lines: List[str] = []
                removed_in_file: List[Dict[str, Any]] = []
                clamped_in_file: List[Dict[str, Any]] = []
                file_changed = False

                for idx, raw_line in enumerate(original_lines, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue

                    cleaned, reason = _clean_label_line(stripped, num_classes=num_classes)

                    if cleaned is None and reason is None:
                        continue

                    if cleaned is None and reason is not None:
                        file_changed = True
                        removed_in_file.append(
                            {
                                "line": idx,
                                "reason": reason,
                                "value": stripped,
                            }
                        )
                        continue

                    if cleaned is not None:
                        new_lines.append(cleaned)
                        if reason == "clamped_box_to_image_bounds":
                            file_changed = True
                            clamped_in_file.append(
                                {
                                    "line": idx,
                                    "reason": reason,
                                    "before": stripped,
                                    "after": cleaned,
                                }
                            )

                if not file_changed:
                    continue

                backup_path = _backup_file_once(label_path, backup_root, dataset_root)
                if backup_path is not None:
                    report["backups_written"] += 1

                new_text = ""
                if new_lines:
                    new_text = "\n".join(new_lines) + "\n"

                with open(label_path, "w", encoding="utf-8") as f:
                    f.write(new_text)

                report["label_files_modified"] += 1
                split_summary["label_files_modified"] += 1

                report["rows_removed"] += len(removed_in_file)
                split_summary["rows_removed"] += len(removed_in_file)

                report["rows_clamped"] += len(clamped_in_file)
                split_summary["rows_clamped"] += len(clamped_in_file)

                report["changes"].append(
                    {
                        "split": split,
                        "file": _safe_relpath(label_path, dataset_root),
                        "action": "rewrote_label_file",
                        "rows_removed": removed_in_file,
                        "rows_clamped": clamped_in_file,
                    }
                )

            report["splits"][split] = split_summary

        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            if backup_root:
                os.makedirs(backup_root, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)

            log_stage(
                LOG,
                "ARTIFACT",
                "DatasetAutoFix",
                autofix_path=report_path,
                backup_root=backup_root,
            )

        log_stage(
            LOG,
            "INFO",
            "DatasetAutoFix",
            files_scanned=report["files_scanned"],
            label_files_modified=report["label_files_modified"],
            rows_removed=report["rows_removed"],
            rows_clamped=report["rows_clamped"],
            orphan_label_files_deleted=report["orphan_label_files_deleted"],
            backups_written=report["backups_written"],
        )

        return report


def clear_yolo_label_caches(raw_dataset: str, LOG) -> int:
    dataset_root = _ensure_dataset_root(raw_dataset)
    removed = 0

    for p in glob.glob(os.path.join(dataset_root, "**", "*.cache"), recursive=True):
        try:
            os.remove(p)
            removed += 1
        except Exception as e:
            log_stage(
                LOG,
                "WARN",
                "DatasetValidation",
                action="cache_remove_failed",
                path=p,
                error=str(e),
            )

    log_stage(
        LOG,
        "INFO",
        "DatasetValidation",
        action="clear_label_caches",
        cache_files_removed=removed,
    )
    return removed


def validate_yolo_detection_dataset(
    raw_dataset: str,
    run_dir: Optional[str],
    LOG,
) -> Dict[str, Any]:
    """
    Validate a YOLO detection dataset before training.

    Important behavior:
      - Missing label files are allowed and treated as background / hard-negative images.
      - Empty label files are allowed and treated as background / hard-negative images.
      - Detection rows must have exactly 5 values.
      - Segmentation-style rows (>5 values) are treated as fatal issues.
    """
    dataset_root = _ensure_dataset_root(raw_dataset)

    with component_stage(LOG, "DatasetValidation", dataset_root=dataset_root, run_dir=run_dir):
        data = _load_data_yaml(dataset_root)
        names = data["names"]

        summary: Dict[str, Any] = {
            "dataset_root": dataset_root,
            "data_yaml": os.path.join(dataset_root, "data.yaml"),
            "class_names": names,
            "num_classes": len(names),
            "splits": {},
            "fatal_issues": [],
            "warning_issues": [],
            "ok": True,
        }

        found_split = False
        for split in SPLITS:
            split_dir = os.path.join(dataset_root, split)
            if not os.path.isdir(split_dir):
                continue

            found_split = True
            split_summary = _scan_split(
                dataset_root=dataset_root,
                split=split,
                num_classes=len(names),
            )

            summary["splits"][split] = {
                "images": split_summary["images"],
                "label_files": split_summary["label_files"],
                "background_images": split_summary["background_images"],
                "empty_label_files": split_summary["empty_label_files"],
                "bad_images": split_summary["bad_images"],
                "bad_label_files": split_summary["bad_label_files"],
            }

            for item in split_summary["issues"]:
                if item["severity"] == "fatal":
                    summary["fatal_issues"].append(item)
                else:
                    summary["warning_issues"].append(item)

        if not found_split:
            raise RuntimeError(f"No dataset splits found under: {dataset_root}")

        summary["fatal_issue_count"] = len(summary["fatal_issues"])
        summary["warning_issue_count"] = len(summary["warning_issues"])
        summary["ok"] = summary["fatal_issue_count"] == 0

        artifact_path = None
        if run_dir:
            artifacts_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            artifact_path = os.path.join(artifacts_dir, "dataset_validation.json")
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4)

            log_stage(
                LOG,
                "ARTIFACT",
                "DatasetValidation",
                validation_path=artifact_path,
            )

        log_stage(
            LOG,
            "INFO" if summary["ok"] else "WARN",
            "DatasetValidation",
            ok=summary["ok"],
            fatal_issue_count=summary["fatal_issue_count"],
            warning_issue_count=summary["warning_issue_count"],
            splits=list(summary["splits"].keys()),
            artifact=artifact_path,
        )

        for split_name, split_info in summary["splits"].items():
            log_stage(
                LOG,
                "INFO",
                "DatasetValidation",
                split=split_name,
                images=split_info["images"],
                label_files=split_info["label_files"],
                background_images=split_info["background_images"],
                empty_label_files=split_info["empty_label_files"],
                bad_images=split_info["bad_images"],
                bad_label_files=split_info["bad_label_files"],
            )

        if summary["fatal_issue_count"] > 0:
            first = summary["fatal_issues"][0]
            log_stage(
                LOG,
                "WARN",
                "DatasetValidation",
                first_fatal_type=first.get("type"),
                first_fatal_file=first.get("file"),
                first_fatal_line=first.get("line"),
            )

        return summary
