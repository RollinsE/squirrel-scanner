import json
import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from src.logger import component_stage, log_stage
from src.train_det_ultra import train_ultralytics_detector
from src.validate_dataset import (
    auto_fix_yolo_detection_dataset,
    clear_yolo_label_caches,
    validate_yolo_detection_dataset,
)


def _ensure_dirs(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _pipeline_state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "artifacts", "pipeline_state.json")


def _read_pipeline_state(run_dir: str) -> Dict[str, Any]:
    state = _read_json(_pipeline_state_path(run_dir))
    default = {
        "version": 1,
        "status": "idle",  # idle | active | completed
        "active_detector_tag": None,
        "target_total_epochs": 0,
        "updated_at_epoch_sec": int(time.time()),
    }
    if state:
        default.update(state)
    return default


def _write_pipeline_state(run_dir: str, state: Dict[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at_epoch_sec"] = int(time.time())
    _write_json(_pipeline_state_path(run_dir), payload)


def _detector_state_path(run_dir: str, tag: str) -> str:
    return os.path.join(run_dir, "artifacts", "detectors", tag, "detector_state.json")


def _detector_existing_result(run_dir: str, tag: str) -> Dict[str, Any]:
    state = _read_json(_detector_state_path(run_dir, tag))
    return {
        "tag": tag,
        "best_weights": state.get("last_stable_best"),
        "score": float(state.get("last_score", 0.0) or 0.0),
        "completed_epochs": int(state.get("effective_completed_epochs", 0) or 0),
        "target_total_epochs": int(state.get("target_total_epochs", 0) or 0),
        "loaded_from_state": True,
    }




def _detector_completed_epochs(run_dir: str, tag: str) -> int:
    state = _read_json(_detector_state_path(run_dir, tag))
    try:
        return int(state.get("effective_completed_epochs", 0) or 0)
    except Exception:
        return 0


def _select_resume_start_index(
    *,
    run_dir: str,
    detector_specs: Sequence[Mapping[str, Any]],
    requested_epochs: int,
    active_tag: Optional[str],
) -> int:
    """Return the first detector that needs work for the requested target epoch count.

    If a run was interrupted, the active detector is normally the best starting point.
    If the requested epoch target is higher than the completed epoch count for any
    earlier detector, start from the earliest detector that still needs work.
    """
    if not detector_specs:
        return 0

    active_idx = None
    for i, spec in enumerate(detector_specs):
        if spec.get("tag") == active_tag:
            active_idx = i
            break

    first_incomplete_idx = None
    for i, spec in enumerate(detector_specs):
        completed = _detector_completed_epochs(run_dir, str(spec.get("tag", "")))
        if completed < requested_epochs:
            first_incomplete_idx = i
            break

    if first_incomplete_idx is None:
        return 0
    if active_idx is None:
        return first_incomplete_idx
    return min(active_idx, first_incomplete_idx)


def _cuda_cleanup(reason: str, LOG) -> None:
    if not torch.cuda.is_available():
        return

    try:
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        max_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

        log_stage(
            LOG,
            "INFO",
            "CUDA",
            action="cleanup_start",
            reason=reason,
            cuda=True,
            allocated_mb=round(allocated, 2),
            reserved_mb=round(reserved, 2),
            max_allocated_mb=round(max_alloc, 2),
            max_reserved_mb=round(max_res, 2),
        )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        allocated2 = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved2 = torch.cuda.memory_reserved() / (1024 * 1024)

        log_stage(
            LOG,
            "INFO",
            "CUDA",
            action="cleanup_done",
            reason=reason,
            cuda=True,
            allocated_mb=round(allocated2, 2),
            reserved_mb=round(reserved2, 2),
        )
    except Exception:
        LOG.exception("CUDA cleanup failed (non-fatal).")


def _score(result: Dict[str, Any]) -> float:
    try:
        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


def parse_model_list(value: Optional[str]) -> List[str]:
    """Parse a comma-separated model list from the CLI."""
    if value is None:
        return ["yolo11n"]
    models = [m.strip() for m in value.split(",") if m.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name or weight path.")
    return models


def parse_model_batches(value: Optional[str]) -> Dict[str, int]:
    """Parse per-model batch overrides, e.g. 'yolo11n=8,rtdetr-l=4'."""
    if not value:
        return {}
    out: Dict[str, int] = {}
    for chunk in re.split(r"[;,]", value):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk and ":" not in chunk:
            raise ValueError("Invalid --model_batches entry. Use model=batch, e.g. yolo11n=8,rtdetr-l=4")
        sep = "=" if "=" in chunk else ":"
        key, raw_val = chunk.split(sep, 1)
        key = key.strip()
        if not key:
            raise ValueError("Invalid --model_batches entry with empty model name.")
        out[_model_tag(key)] = int(raw_val.strip())
    return out


def _model_tag(model_name: str) -> str:
    base = os.path.basename(str(model_name).strip())
    for suffix in (".pt", ".yaml", ".yml"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", base).strip(".-_").lower()
    if not tag:
        raise ValueError(f"Could not derive a detector tag from model name: {model_name}")
    return tag


def _pretrained_from_model_name(model_name: str) -> str:
    value = str(model_name).strip()
    if value.lower().endswith((".pt", ".yaml", ".yml")) or os.path.exists(value):
        return value
    return f"{value}.pt"


def _batch_for_model(
    *,
    tag: str,
    model_name: str,
    model_batches: Mapping[str, int],
    default_batch: int,
    batch_yolo11: int,
    batch_yolo26: int,
    batch_rtdetr: int,
) -> int:
    keys = {_model_tag(model_name), tag}
    for key in keys:
        if key in model_batches:
            return int(model_batches[key])
    if tag.startswith("yolo11"):
        return int(batch_yolo11)
    if tag.startswith("yolo26"):
        return int(batch_yolo26)
    if tag.startswith("rtdetr"):
        return int(batch_rtdetr)
    return int(default_batch)


def _build_detector_specs(
    *,
    models: Sequence[str],
    model_batches: Mapping[str, int],
    default_batch: int,
    batch_yolo11: int,
    batch_yolo26: int,
    batch_rtdetr: int,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    seen_tags = set()
    for model_name in models:
        tag = _model_tag(model_name)
        if tag in seen_tags:
            raise ValueError(f"Duplicate model tag after normalization: {tag}")
        seen_tags.add(tag)
        specs.append(
            {
                "tag": tag,
                "pretrained": _pretrained_from_model_name(model_name),
                "batch": _batch_for_model(
                    tag=tag,
                    model_name=model_name,
                    model_batches=model_batches,
                    default_batch=default_batch,
                    batch_yolo11=batch_yolo11,
                    batch_yolo26=batch_yolo26,
                    batch_rtdetr=batch_rtdetr,
                ),
                "cleanup_reason": f"after_{tag.replace('-', '_')}",
            }
        )
    return specs


def train_detectors(
    raw_dataset: str,
    run_dir: str,
    epochs: int,
    imgsz: int,
    batch_yolo11: int,
    batch_yolo26: int,
    batch_rtdetr: int,
    lr: float,
    num_workers: int,
    ultra_device: str,
    eval_every: int,
    resume: bool,
    LOG,
    models: Optional[Sequence[str]] = None,
    model_batches: Optional[Mapping[str, int]] = None,
    default_batch: int = 8,
    optimizer: str = "auto",
    weight_decay: float = 0.0005,
    momentum: float = 0.937,
) -> dict:
    """Train one or more Ultralytics detectors in the requested order.

    Resume behavior:
      - remembers which detector was active when the run was interrupted
      - resumes that detector first on the next --resume call
    """
    _ensure_dirs(run_dir)

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    requested_models = list(models) if models is not None else parse_model_list(None)
    batch_overrides = dict(model_batches or {})
    detector_specs = _build_detector_specs(
        models=requested_models,
        model_batches=batch_overrides,
        default_batch=default_batch,
        batch_yolo11=batch_yolo11,
        batch_yolo26=batch_yolo26,
        batch_rtdetr=batch_rtdetr,
    )

    with component_stage(
        LOG,
        "DetectorTraining",
        raw_dataset=raw_dataset,
        run_dir=run_dir,
        resume=resume,
        epochs=epochs,
        imgsz=imgsz,
        batch_yolo11=batch_yolo11,
        batch_yolo26=batch_yolo26,
        batch_rtdetr=batch_rtdetr,
        default_batch=default_batch,
        model_batches=batch_overrides,
        models=requested_models,
        lr=lr,
        num_workers=num_workers,
        ultra_device=ultra_device,
        eval_every=eval_every,
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=momentum,
    ):
        log_stage(
            LOG,
            "INFO",
            "DetectorTraining",
            message=f"Starting detector training for {len(detector_specs)} model(s)",
            models=" -> ".join(spec["tag"] for spec in detector_specs),
        )

        fix_report = auto_fix_yolo_detection_dataset(
            raw_dataset=raw_dataset,
            run_dir=run_dir,
            LOG=LOG,
        )

        validation = validate_yolo_detection_dataset(
            raw_dataset=raw_dataset,
            run_dir=run_dir,
            LOG=LOG,
        )

        if not validation.get("ok", False):
            raise RuntimeError(
                "Dataset validation failed after auto-fix. "
                "See artifacts/dataset_validation.json for details."
            )

        clear_yolo_label_caches(raw_dataset, LOG)
        _cuda_cleanup("before_training_start", LOG)

        pipeline_state = _read_pipeline_state(run_dir)
        pipeline_state["target_total_epochs"] = epochs

        active_tag = pipeline_state.get("active_detector_tag") if resume else None
        if resume:
            start_idx = _select_resume_start_index(
                run_dir=run_dir,
                detector_specs=detector_specs,
                requested_epochs=epochs,
                active_tag=active_tag,
            )
        else:
            start_idx = 0

        # Preload earlier detector results only when they already satisfy the requested target.
        for i in range(start_idx):
            tag = detector_specs[i]["tag"]
            completed = _detector_completed_epochs(run_dir, tag)
            if completed < epochs:
                raise RuntimeError(
                    f"Internal resume error: detector '{tag}' has only {completed} completed "
                    f"epoch(s), below requested target {epochs}, but was selected to be skipped."
                )
            results.append(_detector_existing_result(run_dir, tag))

        for i in range(start_idx, len(detector_specs)):
            spec = detector_specs[i]
            tag = spec["tag"]

            pipeline_state["status"] = "active"
            pipeline_state["active_detector_tag"] = tag
            _write_pipeline_state(run_dir, pipeline_state)

            result = train_ultralytics_detector(
                tag=tag,
                pretrained=spec["pretrained"],
                raw_dataset=raw_dataset,
                run_dir=run_dir,
                epochs=epochs,
                imgsz=imgsz,
                batch=spec["batch"],
                device=ultra_device,
                resume=resume,
                lr=lr,
                num_workers=num_workers,
                optimizer=optimizer,
                weight_decay=weight_decay,
                momentum=momentum,
                LOG=LOG,
            )
            results.append(result)

            detector_state = _read_json(_detector_state_path(run_dir, tag))
            detector_state["last_score"] = float(result.get("score", 0.0))
            _write_json(_detector_state_path(run_dir, tag), detector_state)

            _cuda_cleanup(spec["cleanup_reason"], LOG)

        pipeline_state["status"] = "completed"
        pipeline_state["active_detector_tag"] = None
        _write_pipeline_state(run_dir, pipeline_state)

        champion = max(results, key=_score)

        summary = {
            "raw_dataset": raw_dataset,
            "run_dir": run_dir,
            "settings": {
                "epochs": epochs,
                "imgsz": imgsz,
                "batch_yolo11": batch_yolo11,
                "batch_yolo26": batch_yolo26,
                "batch_rtdetr": batch_rtdetr,
                "default_batch": default_batch,
                "model_batches": batch_overrides,
                "models": requested_models,
                "detector_specs": detector_specs,
                "lr": lr,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "num_workers": num_workers,
                "ultra_device": ultra_device,
                "eval_every": eval_every,
                "resume": resume,
            },
            "dataset_autofix": fix_report,
            "dataset_validation": validation,
            "results": results,
            "champion": champion,
            "elapsed_sec": round(time.time() - t0, 2),
        }

        metrics_path = os.path.join(run_dir, "metrics", "detectors_summary.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        champion_path = os.path.join(run_dir, "artifacts", "champion_detector.json")
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(champion, f, indent=4)

        log_stage(
            LOG,
            "ARTIFACT",
            "DetectorTraining",
            detectors_summary=metrics_path,
            champion_detector=champion_path,
            pipeline_state=_pipeline_state_path(run_dir),
        )
        log_stage(
            LOG,
            "RESULT",
            "DetectorTraining",
            champion_tag=champion.get("tag"),
            champion_score=champion.get("score"),
            champion_best_weights=champion.get("best_weights"),
        )

        return summary
