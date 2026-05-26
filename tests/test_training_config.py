from pathlib import Path
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.dataset_merge import merge_yolo_detection_datasets, parse_class_map
from src.train_det import parse_model_batches, parse_model_list


def _make_yolo_dataset(root: Path, names, labels):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "valid" / "images").mkdir(parents=True)
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"train": "train/images", "val": "valid/images", "nc": len(names), "names": names}, sort_keys=False),
        encoding="utf-8",
    )
    for stem, line in labels.items():
        (root / "train" / "images" / f"{stem}.jpg").write_bytes(b"fake-image")
        (root / "train" / "labels" / f"{stem}.txt").write_text(line, encoding="utf-8")


def test_parse_model_list_allows_one_or_many():
    assert parse_model_list(None) == ["yolo11n"]
    assert parse_model_list("yolo11n") == ["yolo11n"]
    assert parse_model_list("yolo11n, yolo11s, rtdetr-l") == ["yolo11n", "yolo11s", "rtdetr-l"]


def test_parse_model_batches_normalizes_weight_suffix():
    assert parse_model_batches("yolo11n.pt=8,rtdetr-l=2") == {"yolo11n": 8, "rtdetr-l": 2}


def test_merge_yolo_detection_datasets_remaps_classes(tmp_path):
    ds1 = tmp_path / "squirrel_3class"
    ds2 = tmp_path / "red_rat"
    ds3 = tmp_path / "rat_only"
    _make_yolo_dataset(ds1, ["red squirrels", "grey squirrels", "martens"], {"a": "0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.1 0.1\n"})
    _make_yolo_dataset(ds2, ["red squirrels", "rats"], {"b": "1 0.5 0.5 0.2 0.2\n"})
    _make_yolo_dataset(ds3, ["rats"], {"c": "0 0.2 0.2 0.1 0.1\n"})

    merged = merge_yolo_detection_datasets(
        dataset_roots=[str(ds1), str(ds2), str(ds3)],
        output_dir=str(tmp_path / "merged"),
        target_classes=["red", "grey", "marten", "rat"],
        class_map=parse_class_map("red squirrels=red,grey squirrels=grey,martens=marten,rats=rat"),
    )

    merged_root = Path(merged)
    data = yaml.safe_load((merged_root / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == ["red", "grey", "marten", "rat"]

    label_text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((merged_root / "train" / "labels").glob("*.txt")))
    assert "0 0.5 0.5 0.1 0.1" in label_text
    assert "1 0.4 0.4 0.1 0.1" in label_text
    assert "3 0.5 0.5 0.2 0.2" in label_text
    assert "3 0.2 0.2 0.1 0.1" in label_text



def test_merge_rejects_unmapped_source_classes(tmp_path):
    ds = tmp_path / "unexpected"
    _make_yolo_dataset(ds, ["brown squirrel"], {"a": "0 0.5 0.5 0.1 0.1\n"})

    try:
        merge_yolo_detection_datasets(
            dataset_roots=[str(ds)],
            output_dir=str(tmp_path / "merged"),
            target_classes=["red", "grey", "marten", "rat"],
            class_map=parse_class_map(""),
        )
    except ValueError as exc:
        message = str(exc)
        assert "brown squirrel" in message
        assert "--class_map" in message
    else:
        raise AssertionError("Expected merge to reject an unmapped class name.")



def test_merge_replaces_existing_output_folder(tmp_path):
    ds = tmp_path / "source"
    _make_yolo_dataset(ds, ["red"], {"a": "0 0.5 0.5 0.1 0.1\n"})
    out = tmp_path / "merged"
    (out / "train" / "labels").mkdir(parents=True)
    (out / "train" / "labels" / "stale.txt").write_text("stale", encoding="utf-8")

    merged = merge_yolo_detection_datasets(
        dataset_roots=[str(ds)],
        output_dir=str(out),
        target_classes=["red"],
    )

    assert Path(merged).exists()
    assert not (out / "train" / "labels" / "stale.txt").exists()


from src.train_det import _select_resume_start_index
import json


def _write_detector_state(root: Path, tag: str, completed_epochs: int):
    state_dir = root / "artifacts" / "detectors" / tag
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "detector_state.json").write_text(
        json.dumps({"effective_completed_epochs": completed_epochs}),
        encoding="utf-8",
    )


def test_resume_epoch_increase_restarts_from_earlier_incomplete_model(tmp_path):
    _write_detector_state(tmp_path, "yolo11n", 60)
    _write_detector_state(tmp_path, "yolo11s", 5)

    specs = [{"tag": "yolo11n"}, {"tag": "yolo11s"}]
    assert _select_resume_start_index(
        run_dir=str(tmp_path),
        detector_specs=specs,
        requested_epochs=80,
        active_tag="yolo11s",
    ) == 0


def test_resume_same_epoch_starts_from_interrupted_active_model(tmp_path):
    _write_detector_state(tmp_path, "yolo11n", 60)
    _write_detector_state(tmp_path, "yolo11s", 5)

    specs = [{"tag": "yolo11n"}, {"tag": "yolo11s"}]
    assert _select_resume_start_index(
        run_dir=str(tmp_path),
        detector_specs=specs,
        requested_epochs=60,
        active_tag="yolo11s",
    ) == 1


def test_resume_lower_epoch_target_detects_all_models_are_complete(tmp_path):
    _write_detector_state(tmp_path, "yolo11n", 60)
    _write_detector_state(tmp_path, "yolo11s", 80)

    specs = [{"tag": "yolo11n"}, {"tag": "yolo11s"}]
    assert _select_resume_start_index(
        run_dir=str(tmp_path),
        detector_specs=specs,
        requested_epochs=50,
        active_tag="yolo11s",
    ) == 0
