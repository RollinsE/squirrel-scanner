import os
import glob
import shutil
import yaml
from collections import Counter


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _find_yolo_yaml(dataset_root: str) -> str:
    """Find a YOLO dataset YAML that contains 'names'."""
    direct = os.path.join(dataset_root, "data.yaml")
    if os.path.exists(direct):
        return direct

    candidates = glob.glob(os.path.join(dataset_root, "**", "*.yaml"), recursive=True)
    candidates += glob.glob(os.path.join(dataset_root, "**", "*.yml"), recursive=True)

    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if isinstance(d, dict) and "names" in d:
                return p
        except Exception:
            continue

    raise FileNotFoundError(f"Could not find a YOLO yaml (with 'names') under: {dataset_root}")


def _find_yolo_root_from_yaml(yaml_path: str) -> str:
    """YOLO yaml is usually at dataset root; return its directory."""
    return os.path.dirname(yaml_path)


def _list_images(images_dir: str, stem: str) -> list[str]:
    hits = []
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            hits.append(p)
    if not hits:
        hits = glob.glob(os.path.join(images_dir, f"{stem}.*"))
    return hits


def _count_classification_split(split_dir: str) -> dict:
    """Return per-class counts for ImageFolder split."""
    out = {}
    if not os.path.isdir(split_dir):
        return out
    for cls in sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]):
        out[cls] = len([f for f in os.listdir(os.path.join(split_dir, cls)) if f.lower().endswith(IMG_EXTS)])
    return out


def convert_yolo_detection_to_imagefolder(dataset_root: str, LOG):
    """
    Convert YOLO detection dataset into ImageFolder classification dataset.

    Rule:
      - For each label file, take the first object's class_id and assign the image to that class.
      - Empty label files are skipped.

    Writes to: <parent_of_dataset_root>/classification_dataset
    """
    LOG.info("[PREPROCESS] START model=YOLO->ImageFolder")
    LOG.info(f"[PREPROCESS] dataset_root={dataset_root}")

    yaml_path = _find_yolo_yaml(dataset_root)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)

    classes = data_yaml.get("names", [])
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"Invalid 'names' in yaml: {yaml_path}")

    LOG.info(f"[PREPROCESS] yaml_path={yaml_path}")
    LOG.info(f"[PREPROCESS] num_classes={len(classes)} classes={classes}")

    yolo_root = _find_yolo_root_from_yaml(yaml_path)
    parent = os.path.dirname(yolo_root)
    cls_root = os.path.join(parent, "classification_dataset")
    os.makedirs(cls_root, exist_ok=True)

    for split in ("train", "valid", "test"):
        for cls_name in classes:
            os.makedirs(os.path.join(cls_root, split, str(cls_name)), exist_ok=True)

    copied = 0
    skipped_empty = 0
    skipped_missing_img = 0
    bad_labels = 0
    per_split_counter = {"train": Counter(), "valid": Counter(), "test": Counter()}

    for split in ("train", "valid", "test"):
        split_dir = os.path.join(yolo_root, split)
        if not os.path.isdir(split_dir):
            LOG.warning(f"[PREPROCESS] split_missing split={split} path={split_dir}")
            continue

        labels_dir = os.path.join(split_dir, "labels")
        images_dir = os.path.join(split_dir, "images")
        if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
            LOG.warning(f"[PREPROCESS] split_incomplete split={split} labels_dir={labels_dir} images_dir={images_dir}")
            continue

        label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
        LOG.info(f"[PREPROCESS] split={split} label_files={len(label_files)}")

        for lf in label_files:
            with open(lf, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            if not lines:
                skipped_empty += 1
                continue

            try:
                class_id = int(lines[0].split()[0])
                class_name = str(classes[class_id])
            except Exception:
                bad_labels += 1
                continue

            stem = os.path.splitext(os.path.basename(lf))[0]
            imgs = _list_images(images_dir, stem)
            if not imgs:
                skipped_missing_img += 1
                continue

            src = imgs[0]
            dst = os.path.join(cls_root, split, class_name, os.path.basename(src))
            shutil.copy2(src, dst)
            copied += 1
            per_split_counter[split][class_name] += 1

    LOG.info(f"[PREPROCESS] DONE copied={copied} skipped_empty={skipped_empty} skipped_missing_img={skipped_missing_img} bad_labels={bad_labels}")

    for split in ("train", "valid", "test"):
        if per_split_counter[split]:
            LOG.info(f"[PREPROCESS] split={split} class_counts={dict(per_split_counter[split])}")

    for split in ("train", "valid", "test"):
        split_counts = _count_classification_split(os.path.join(cls_root, split))
        if split_counts:
            LOG.info(f"[PREPROCESS] imagefolder split={split} counts={split_counts}")

    LOG.info(f"[PREPROCESS] classification_dataset_root={cls_root}")
    return cls_root
