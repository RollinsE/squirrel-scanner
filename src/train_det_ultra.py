import csv
import json
import os
import shutil
import time
from typing import Any, Dict, Optional

import torch

from src.logger import component_stage, log_stage


def _ensure_dataset_root(raw_dataset: str) -> str:
    if not raw_dataset or not os.path.isdir(raw_dataset):
        raise FileNotFoundError(f"--raw_dataset is not a directory: {raw_dataset}")
    data_yaml = os.path.join(raw_dataset, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Expected data.yaml at: {data_yaml}")
    return raw_dataset


def _results_csv_path(save_dir: str) -> str:
    return os.path.join(save_dir, "results.csv")


def _read_last_epoch_from_results(save_dir: str) -> Optional[int]:
    """
    Ultralytics results.csv uses 0-based epoch numbering.
    Return completed epochs as a human 1-based count, or None if unavailable.
    """
    path = _results_csv_path(save_dir)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row

        if not last_row:
            return None

        raw_epoch = last_row.get("epoch")
        if raw_epoch is None:
            return None

        return int(float(raw_epoch)) + 1
    except Exception:
        return None


def _read_last_map50_95_from_results(save_dir: str) -> Optional[float]:
    path = _results_csv_path(save_dir)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row

        if not last_row:
            return None

        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(box)"):
            value = last_row.get(key)
            if value not in ("", None):
                return float(value)

        return None
    except Exception:
        return None


def _extract_map50_95_from_results(results_obj: Any, fallback_dir: str) -> float:
    value = None
    try:
        results_dict = getattr(results_obj, "results_dict", None) or {}
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(box)"):
            if key in results_dict:
                value = float(results_dict[key])
                break
    except Exception:
        value = None

    if value is None:
        value = _read_last_map50_95_from_results(fallback_dir)

    return float(value) if value is not None else 0.0


def _copy_file(src: str, dst: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


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


def _extract_checkpoint_field(ckpt_path: str, field: str) -> Optional[int]:
    if not os.path.isfile(ckpt_path):
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            return None

        if field == "epoch":
            epoch = ckpt.get("epoch", None)
            return int(epoch) + 1 if epoch is not None else None

        for key in ("train_args", "args"):
            obj = ckpt.get(key, None)
            if obj is None:
                continue

            if isinstance(obj, dict) and field in obj:
                return int(obj[field])

            if hasattr(obj, field):
                return int(getattr(obj, field))

        return None
    except Exception:
        return None


def _extract_checkpoint_completed_epoch(ckpt_path: str) -> Optional[int]:
    return _extract_checkpoint_field(ckpt_path, "epoch")


def _extract_checkpoint_planned_epochs(ckpt_path: str) -> Optional[int]:
    return _extract_checkpoint_field(ckpt_path, "epochs")


def _build_resume_shadow_callback(resume_shadow_pt: str, LOG, tag: str):
    """
    Copy trainer.last -> canonical resume_last.pt on every save so interrupted
    runs can be resumed in place.
    """

    def _callback(trainer):
        try:
            src = str(trainer.last)
            if os.path.isfile(src):
                ok = _copy_file(src, resume_shadow_pt)
                if not ok:
                    log_stage(
                        LOG,
                        "WARN",
                        tag,
                        message="Failed to copy resumable shadow checkpoint on save.",
                        src=src,
                        dst=resume_shadow_pt,
                    )
        except Exception as e:
            log_stage(
                LOG,
                "WARN",
                tag,
                message="Exception while copying resumable shadow checkpoint.",
                error=str(e),
                dst=resume_shadow_pt,
            )

    return _callback


def _detector_paths(det_dir: str) -> Dict[str, str]:
    train_dir = os.path.join(det_dir, "train")
    weights_dir = os.path.join(train_dir, "weights")
    return {
        "det_dir": det_dir,
        "train_dir": train_dir,
        "weights_dir": weights_dir,
        "best_pt": os.path.join(weights_dir, "best.pt"),
        "last_pt": os.path.join(weights_dir, "last.pt"),
        "resume_last_pt": os.path.join(weights_dir, "resume_last.pt"),
        "state_json": os.path.join(det_dir, "detector_state.json"),
    }


def _ensure_detector_dirs(paths: Dict[str, str]) -> None:
    os.makedirs(paths["det_dir"], exist_ok=True)
    os.makedirs(paths["train_dir"], exist_ok=True)
    os.makedirs(paths["weights_dir"], exist_ok=True)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _default_detector_state(paths: Dict[str, str]) -> Dict[str, Any]:
    return {
        "version": 4,
        "status": "idle",  # idle | active | completed
        "effective_completed_epochs": 0,
        "target_total_epochs": 0,
        "last_stable_best": paths["best_pt"] if os.path.isfile(paths["best_pt"]) else None,
        "last_resumable_checkpoint": (
            paths["resume_last_pt"]
            if os.path.isfile(paths["resume_last_pt"])
            else (paths["last_pt"] if os.path.isfile(paths["last_pt"]) else None)
        ),
        "active_phase": None,
        "updated_at_epoch_sec": int(time.time()),
    }


def _read_detector_state(paths: Dict[str, str]) -> Dict[str, Any]:
    state = _default_detector_state(paths)
    existing = _read_json(paths["state_json"])
    if existing:
        state.update(existing)

    if not state.get("last_stable_best") and os.path.isfile(paths["best_pt"]):
        state["last_stable_best"] = paths["best_pt"]

    if not state.get("last_resumable_checkpoint"):
        if os.path.isfile(paths["resume_last_pt"]):
            state["last_resumable_checkpoint"] = paths["resume_last_pt"]
        elif os.path.isfile(paths["last_pt"]):
            state["last_resumable_checkpoint"] = paths["last_pt"]

    return state


def _write_detector_state(paths: Dict[str, str], state: Dict[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at_epoch_sec"] = int(time.time())
    _write_json(paths["state_json"], payload)


def _canonical_raw_local_epochs(paths: Dict[str, str]) -> int:
    """
    Raw local epochs visible in canonical train artifacts.
    This value is only safe to use together with detector state / active_phase.
    """
    csv_done = _read_last_epoch_from_results(paths["train_dir"]) or 0
    resume_done = _extract_checkpoint_completed_epoch(paths["resume_last_pt"]) or 0
    last_done = _extract_checkpoint_completed_epoch(paths["last_pt"]) or 0
    return max(csv_done, resume_done, last_done)


def _reconcile_state_with_artifacts(paths: Dict[str, str], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep state in sync without destroying cumulative extension progress.
    """
    state = dict(state)
    raw_local = _canonical_raw_local_epochs(paths)
    target_total = _safe_int(state.get("target_total_epochs"), 0)
    active_phase = state.get("active_phase") or None

    if active_phase:
        base_completed = _safe_int(active_phase.get("base_completed_epochs"), 0)
        planned_local = _safe_int(active_phase.get("planned_local_epochs"), 0)
        target_for_phase = _safe_int(active_phase.get("target_total_epochs"), target_total)

        local_done = min(max(raw_local, 0), max(planned_local, 0))
        active_phase["local_completed_epochs"] = local_done

        effective = base_completed + local_done
        if planned_local > 0 and local_done >= planned_local:
            effective = target_for_phase
            active_phase["status"] = "completed"
            state["status"] = "completed"
            state["active_phase"] = None
        else:
            state["status"] = "active"
            state["active_phase"] = active_phase

        state["effective_completed_epochs"] = max(
            _safe_int(state.get("effective_completed_epochs"), 0),
            effective,
        )
    else:
        existing_effective = _safe_int(state.get("effective_completed_epochs"), 0)
        if target_total > 0 and existing_effective >= target_total:
            state["effective_completed_epochs"] = target_total
            state["status"] = "completed"
        else:
            # For a plain interrupted base run, raw local is cumulative.
            state["effective_completed_epochs"] = max(existing_effective, raw_local)
            if target_total > 0 and state["effective_completed_epochs"] >= target_total:
                state["effective_completed_epochs"] = target_total
                state["status"] = "completed"

    if os.path.isfile(paths["best_pt"]):
        state["last_stable_best"] = paths["best_pt"]

    if os.path.isfile(paths["resume_last_pt"]):
        state["last_resumable_checkpoint"] = paths["resume_last_pt"]
    elif os.path.isfile(paths["last_pt"]):
        state["last_resumable_checkpoint"] = paths["last_pt"]

    if target_total > 0 and _safe_int(state.get("effective_completed_epochs"), 0) >= target_total:
        state["effective_completed_epochs"] = target_total
        state["status"] = "completed"
        state["active_phase"] = None

    return state


def _prepare_true_resume_phase(state: Dict[str, Any], target_epochs: int) -> Dict[str, Any]:
    state = dict(state)
    current_done = _safe_int(state.get("effective_completed_epochs"), 0)
    state["target_total_epochs"] = target_epochs
    state["status"] = "active"
    state["active_phase"] = {
        "kind": "true_resume",
        "status": "active",
        "base_completed_epochs": 0,
        "planned_local_epochs": target_epochs,
        "local_completed_epochs": current_done,
        "target_total_epochs": target_epochs,
        "start_checkpoint": state.get("last_resumable_checkpoint"),
    }
    return state


def _prepare_extension_phase(state: Dict[str, Any], target_epochs: int, start_weights: str) -> Dict[str, Any]:
    state = dict(state)
    completed = _safe_int(state.get("effective_completed_epochs"), 0)
    state["target_total_epochs"] = target_epochs
    state["status"] = "active"
    state["active_phase"] = {
        "kind": "extension",
        "status": "active",
        "base_completed_epochs": completed,
        "planned_local_epochs": max(0, target_epochs - completed),
        "local_completed_epochs": 0,
        "target_total_epochs": target_epochs,
        "start_checkpoint": start_weights,
    }
    return state


def _mark_phase_completed(paths: Dict[str, str], state: Dict[str, Any], final_target: int) -> Dict[str, Any]:
    state = dict(state)
    state["effective_completed_epochs"] = final_target
    state["target_total_epochs"] = final_target
    state["status"] = "completed"
    state["active_phase"] = None
    if os.path.isfile(paths["best_pt"]):
        state["last_stable_best"] = paths["best_pt"]
    if os.path.isfile(paths["resume_last_pt"]):
        state["last_resumable_checkpoint"] = paths["resume_last_pt"]
    elif os.path.isfile(paths["last_pt"]):
        state["last_resumable_checkpoint"] = paths["last_pt"]
    return state


def train_ultralytics_detector(
    tag: str,
    pretrained: str,
    raw_dataset: str,
    run_dir: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    resume: bool,
    lr: float,
    num_workers: int,
    optimizer: str,
    weight_decay: float,
    momentum: float,
    LOG,
) -> dict:
    """
    Canonical in-place Ultralytics trainer.

    Expected behavior:
      - one canonical detector run dir: artifacts/detectors/<tag>/train
      - interrupted run resumes in place
      - extension after completion stays in the same logical detector run
      - cumulative detector progress is tracked in detector_state.json
    """
    from ultralytics import YOLO

    RTDETR = None
    try:
        from ultralytics import RTDETR as _RTDETR  # type: ignore
        RTDETR = _RTDETR
    except Exception:
        RTDETR = None

    dataset_root = _ensure_dataset_root(raw_dataset)
    data_yaml = os.path.join(dataset_root, "data.yaml")

    det_dir = os.path.join(run_dir, "artifacts", "detectors", tag)
    paths = _detector_paths(det_dir)
    _ensure_detector_dirs(paths)

    with component_stage(
        LOG,
        tag,
        pretrained=pretrained,
        dataset_root=dataset_root,
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        resume=resume,
        lr=lr,
        num_workers=num_workers,
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=momentum,
        last_pt=paths["last_pt"],
        resume_last_pt=paths["resume_last_pt"],
        state_json=paths["state_json"],
    ):
        pretrained_name = os.path.basename(str(pretrained)).lower()
        is_rtdetr = tag.lower().startswith("rtdetr") or pretrained_name.startswith("rtdetr")
        if is_rtdetr:
            if RTDETR is None:
                raise RuntimeError(
                    "RT-DETR requested but your ultralytics package does not expose RTDETR(). "
                    "Fix: upgrade ultralytics (e.g. pip install -U ultralytics)."
                )
            model_cls = RTDETR
        else:
            model_cls = YOLO

        state = _read_detector_state(paths)
        state["target_total_epochs"] = epochs
        state = _reconcile_state_with_artifacts(paths, state)
        _write_detector_state(paths, state)

        completed = _safe_int(state.get("effective_completed_epochs"), 0)

        if completed >= epochs:
            chosen = state.get("last_stable_best") or paths["best_pt"] or paths["last_pt"]
            score = _read_last_map50_95_from_results(paths["train_dir"])
            log_stage(
                LOG,
                "INFO",
                tag,
                message="Requested epoch target is already satisfied; skipping training.",
                completed_epochs=completed,
                requested_epochs=epochs,
                weights=chosen,
                score=score,
            )
            return {
                "tag": tag,
                "best_weights": chosen,
                "score": float(score) if score is not None else 0.0,
                "completed_epochs": completed,
                "target_total_epochs": epochs,
                "skipped": True,
            }

        resumable_ckpt = None
        if os.path.isfile(paths["resume_last_pt"]):
            resumable_ckpt = paths["resume_last_pt"]
        elif os.path.isfile(paths["last_pt"]):
            resumable_ckpt = paths["last_pt"]

        checkpoint_planned = _extract_checkpoint_planned_epochs(resumable_ckpt) if resumable_ckpt else None
        active_phase = state.get("active_phase") or None

        use_true_resume = bool(
            resume
            and resumable_ckpt
            and active_phase
            and active_phase.get("kind") == "true_resume"
            and active_phase.get("status") == "active"
            and _safe_int(active_phase.get("target_total_epochs"), 0) == epochs
        )

        if (
            not use_true_resume
            and resume
            and resumable_ckpt
            and not active_phase
            and checkpoint_planned is not None
            and completed < checkpoint_planned
            and epochs <= checkpoint_planned
        ):
            use_true_resume = True
            state = _prepare_true_resume_phase(state, epochs)
            _write_detector_state(paths, state)

        t0 = time.time()

        if use_true_resume:
            model = model_cls(resumable_ckpt)

            log_stage(
                LOG,
                "INFO",
                tag,
                message="Resuming interrupted canonical run in place.",
                checkpoint=resumable_ckpt,
                completed_epochs=completed,
                requested_epochs=epochs,
                checkpoint_planned_epochs=checkpoint_planned,
                train_dir=paths["train_dir"],
            )

            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                workers=num_workers,
                optimizer=optimizer,
                lr0=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                project=det_dir,
                name="train",
                exist_ok=True,
                resume=True,
            )

            state = _mark_phase_completed(paths, state, epochs)
            _write_detector_state(paths, state)

        else:
            start_weights = state.get("last_stable_best")
            if not start_weights or not os.path.isfile(start_weights):
                if os.path.isfile(paths["best_pt"]):
                    start_weights = paths["best_pt"]
                elif os.path.isfile(paths["last_pt"]):
                    start_weights = paths["last_pt"]
                else:
                    start_weights = pretrained

            state = _prepare_extension_phase(state, epochs, start_weights)
            _write_detector_state(paths, state)

            planned_local = _safe_int(state["active_phase"]["planned_local_epochs"], 0)
            base_completed = _safe_int(state["active_phase"]["base_completed_epochs"], 0)

            model = model_cls(start_weights)
            model.add_callback("on_model_save", _build_resume_shadow_callback(paths["resume_last_pt"], LOG, tag))

            phase = "fresh_start" if base_completed == 0 else "in_place_extension"

            log_stage(
                LOG,
                "INFO",
                tag,
                message="Starting canonical in-place training phase.",
                phase=phase,
                start_weights=start_weights,
                completed_epochs=base_completed,
                requested_total_epochs=epochs,
                planned_local_epochs=planned_local,
                train_dir=paths["train_dir"],
            )

            results = model.train(
                data=data_yaml,
                epochs=planned_local,
                imgsz=imgsz,
                batch=batch,
                device=device,
                workers=num_workers,
                optimizer=optimizer,
                lr0=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                project=det_dir,
                name="train",
                exist_ok=True,
                resume=False,
            )

            state = _mark_phase_completed(paths, state, epochs)
            _write_detector_state(paths, state)

        state = _reconcile_state_with_artifacts(paths, state)
        if _safe_int(state.get("effective_completed_epochs"), 0) > epochs:
            state["effective_completed_epochs"] = epochs
        if _safe_int(state.get("target_total_epochs"), 0) != epochs:
            state["target_total_epochs"] = epochs
        if _safe_int(state.get("effective_completed_epochs"), 0) >= epochs:
            state["status"] = "completed"
            state["active_phase"] = None
        _write_detector_state(paths, state)

        score = _extract_map50_95_from_results(results, paths["train_dir"])
        chosen_best = state.get("last_stable_best") or paths["best_pt"] or paths["last_pt"]

        log_stage(
            LOG,
            "RESULT",
            tag,
            best=chosen_best,
            score=score,
            completed_epochs=state.get("effective_completed_epochs"),
            target_total_epochs=state.get("target_total_epochs"),
            state_json=paths["state_json"],
        )
        log_stage(LOG, "DONE", tag, elapsed_sec=round(time.time() - t0, 1))

        return {
            "tag": tag,
            "best_weights": chosen_best,
            "score": score,
        }
