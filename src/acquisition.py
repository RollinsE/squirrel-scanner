import os
import glob
import json
import re
from datetime import datetime
from typing import Optional

from src.logger import log_stage, component_stage


_MARKERS = (
    "data.yaml",
    "README.roboflow.txt",
    "README.dataset.txt",
    "train/images",
    "train/labels",
    "valid/images",
    "valid/labels",
)


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _ls_some(path: str, max_items: int = 50) -> list[str]:
    try:
        out = []
        for i, e in enumerate(os.scandir(path)):
            if i >= max_items:
                break
            out.append(e.name + ("/" if e.is_dir() else ""))
        return out
    except Exception:
        return []


def _exists_marker(root: str) -> bool:
    for m in _MARKERS:
        if os.path.exists(os.path.join(root, m)):
            return True
    return False


def _find_dataset_root(parent: str) -> Optional[str]:
    """
    Find a folder that looks like a Roboflow YOLO export.
    """
    if not parent or not os.path.exists(parent):
        return None

    # If parent itself looks like a dataset root
    if _exists_marker(parent):
        return parent

    # Otherwise search below it for data.yaml or marker folders
    yaml_hits = glob.glob(os.path.join(parent, "**", "data.yaml"), recursive=True)
    if yaml_hits:
        return os.path.dirname(yaml_hits[0])

    train_imgs = glob.glob(os.path.join(parent, "**", "train", "images"), recursive=True)
    if train_imgs:
        # train/images -> dataset root is two levels up
        return os.path.dirname(os.path.dirname(train_imgs[0]))

    return None


def _count_files(root: str, patterns: list[str]) -> int:
    total = 0
    for pat in patterns:
        total += len(glob.glob(os.path.join(root, pat)))
    return total


def _dataset_stats(root: str) -> dict:
    """
    Simple dataset summary: counts of images/labels per split.
    """
    stats = {"root": root, "splits": {}}
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(root, split, "images")
        lab_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        imgs = _count_files(img_dir, ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"])
        labs = _count_files(lab_dir, ["*.txt"]) if os.path.isdir(lab_dir) else 0
        stats["splits"][split] = {"images": imgs, "labels": labs}
    return stats


def acquire_dataset_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    rf_format: str,
    download_dir: str,
    run_dir: Optional[str],
    LOG
) -> str:
    """
    Downloads a Roboflow dataset and returns the resolved YOLO dataset root.

    Behavior:
      1) Attempt download into an explicit target folder under download_dir.
      2) If that folder has no dataset artifacts, retry with Roboflow default path (no location).
      3) Save acquisition metadata JSON into run_dir/artifacts/acquisition.json when run_dir is provided.

    Returns:
      dataset_root (contains data.yaml and train/valid/test folders)
    """
    # Import here so non-acquire modes don't require roboflow installed
    from roboflow import Roboflow

    os.makedirs(download_dir, exist_ok=True)

    with component_stage(
        LOG,
        "Acquisition",
        workspace=workspace,
        project=project,
        version=version,
        format=rf_format,
        download_dir=download_dir,
    ):
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        ver = proj.version(version)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target_parent = os.path.join(download_dir, f"{_safe_name(project)}_v{version}_{ts}")
        os.makedirs(target_parent, exist_ok=True)

        meta = {
            "workspace": workspace,
            "project": project,
            "version": version,
            "rf_format": rf_format,
            "download_dir": download_dir,
            "attempt_target_parent": target_parent,
            "attempt_1_location_returned": None,
            "attempt_1_target_contents": None,
            "attempt_2_location_returned": None,
            "attempt_2_returned_contents": None,
            "resolved_dataset_root": None,
            "utc_timestamp": ts,
        }

        # Attempt 1: explicit location
        log_stage(LOG, "INFO", "Acquisition", attempt=1, action="download", target_parent=target_parent)
        ds = ver.download(rf_format, location=target_parent)
        returned1 = getattr(ds, "location", None)
        meta["attempt_1_location_returned"] = returned1
        meta["attempt_1_target_contents"] = _ls_some(target_parent)

        log_stage(LOG, "INFO", "Acquisition", attempt=1, returned_location=returned1, ls_target=meta["attempt_1_target_contents"])

        resolved = _find_dataset_root(target_parent) or (_find_dataset_root(returned1) if returned1 else None)

        # Attempt 2: no location (Roboflow default)
        if not resolved:
            log_stage(LOG, "WARN", "Acquisition", attempt=1, reason="no_dataset_artifacts", retry="no_location")
            ds2 = ver.download(rf_format)
            returned2 = getattr(ds2, "location", None)
            meta["attempt_2_location_returned"] = returned2
            meta["attempt_2_returned_contents"] = _ls_some(returned2) if returned2 else None

            log_stage(LOG, "INFO", "Acquisition", attempt=2, returned_location=returned2, ls_returned=meta["attempt_2_returned_contents"])
            resolved = _find_dataset_root(returned2) if returned2 else None

        if not resolved:
            log_stage(LOG, "FAIL", "Acquisition", reason="could_not_locate_dataset_root")
            log_stage(LOG, "INFO", "Acquisition", debug_target_parent=target_parent, ls_target=_ls_some(target_parent))
            if returned1:
                log_stage(LOG, "INFO", "Acquisition", debug_returned1=returned1, ls_returned1=_ls_some(returned1))
            if meta.get("attempt_2_location_returned"):
                r2 = meta["attempt_2_location_returned"]
                log_stage(LOG, "INFO", "Acquisition", debug_returned2=r2, ls_returned2=_ls_some(r2))
            raise RuntimeError("Roboflow download finished but dataset root could not be located.")

        meta["resolved_dataset_root"] = resolved
        stats = _dataset_stats(resolved)

        log_stage(LOG, "INFO", "Acquisition", resolved_dataset_root=resolved)
        log_stage(LOG, "INFO", "Acquisition", dataset_stats=stats)

        # Save metadata
        if run_dir:
            artifacts_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            out_path = os.path.join(artifacts_dir, "acquisition.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({**meta, **stats}, f, indent=4)
            log_stage(LOG, "ARTIFACT", "Acquisition", metadata_path=out_path)

        return resolved
