import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


def _list_videos(video_dir: str) -> List[Path]:
    p = Path(video_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"--video_dir is not a directory: {video_dir}")
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(p.glob(f"*{ext}"))
        vids.extend(p.glob(f"*{ext.upper()}"))
    vids = sorted(list({v.resolve() for v in vids}))
    return vids


def _load_champion(detector_run_dir: str) -> Dict[str, Any]:
    champ_path = Path(detector_run_dir) / "artifacts" / "champion_detector.json"
    if not champ_path.exists():
        raise FileNotFoundError(f"champion_detector.json not found at: {champ_path}")
    with open(champ_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_detector_type(detector_path: str, detector_type: str = "auto") -> str:
    requested = (detector_type or "auto").strip().lower()
    if requested and requested != "auto":
        aliases = {
            "yolo": "yolo",
            "yolo12": "yolo",
            "yolo12n": "yolo",
            "yolo26": "yolo",
            "yolo26n": "yolo",
            "yolov8": "yolo",
            "yolov8n": "yolo",
            "rt-detr": "rtdetr",
            "rtdetr": "rtdetr",
            "rtdetr-l": "rtdetr",
        }
        if requested not in aliases:
            raise ValueError(f"Unsupported detector_type: {detector_type}")
        return aliases[requested]

    name = Path(detector_path).name.lower()
    suffix = Path(detector_path).suffix.lower()

    if "rtdetr" in name or "rt-detr" in name:
        return "rtdetr"
    if "yolo" in name or "yolov" in name or suffix == ".pt":
        return "yolo"

    raise ValueError(
        "Could not auto-detect detector type from the selected file. "
        "Please choose Detector type manually."
    )


def _resolve_detector_source(
    detector_run_dir: Optional[str],
    detector_path: Optional[str],
    detector_type: str,
) -> Dict[str, Any]:
    if detector_path:
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Detector file not found: {detector_path}")

        resolved_type = _infer_detector_type(detector_path, detector_type)
        name = Path(detector_path).stem.lower()

        if resolved_type == "rtdetr":
            tag = "rtdetr-l"
        elif "yolo12" in name:
            tag = "yolo12n"
        elif "yolo26" in name:
            tag = "yolo26n"
        else:
            tag = "yolo"

        return {
            "tag": tag,
            "best_weights": detector_path,
            "source_mode": "detector_path",
            "detector_type": resolved_type,
            "detector_path": detector_path,
            "display_name": Path(detector_path).name,
        }

    if detector_run_dir:
        champ = _load_champion(detector_run_dir)
        champ["source_mode"] = "detector_run_dir"
        champ["detector_run_dir"] = detector_run_dir
        champ["display_name"] = Path(str(champ.get("best_weights", ""))).name or "champion"
        return champ

    raise ValueError("Either detector_run_dir or detector_path must be provided.")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_ultralytics_model(best_weights: str):
    from ultralytics import YOLO
    return YOLO(best_weights)


def _resolve_rescue_conf(primary_conf: float, rescue_conf: Optional[float]) -> float:
    if rescue_conf is not None:
        return float(rescue_conf)
    return max(0.15, min(float(primary_conf), 0.35))


def _resolve_rescue_frame_skip(primary_frame_skip: int, rescue_frame_skip: Optional[int]) -> int:
    if rescue_frame_skip is not None:
        return max(1, int(rescue_frame_skip))
    if primary_frame_skip <= 1:
        return 1
    return max(1, primary_frame_skip // 3)


def _safe_int_or_none(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    iv = int(value)
    return iv if iv > 0 else None


def _is_cancelled(cancel_event) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _passes_box_filters(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    max_box_aspect_ratio: float,
) -> bool:
    bw = max(1.0, float(x2) - float(x1))
    bh = max(1.0, float(y2) - float(y1))
    frame_area = max(1.0, float(frame_w * frame_h))
    area_ratio = (bw * bh) / frame_area
    aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))

    if area_ratio < float(min_box_area_ratio):
        return False
    if area_ratio > float(max_box_area_ratio):
        return False
    if aspect_ratio > float(max_box_aspect_ratio):
        return False
    return True


def _edge_flags(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
    edge_margin_ratio: float,
) -> Tuple[bool, List[str]]:
    mx = frame_w * float(edge_margin_ratio)
    my = frame_h * float(edge_margin_ratio)
    reasons = []
    if x1 <= mx:
        reasons.append("edge_left")
    if y1 <= my:
        reasons.append("edge_top")
    if x2 >= (frame_w - mx):
        reasons.append("edge_right")
    if y2 >= (frame_h - my):
        reasons.append("edge_bottom")
    return len(reasons) > 0, reasons


def _blur_score_from_crop(frame, x1: float, y1: float, x2: float, y2: float) -> float:
    h, w = frame.shape[:2]
    ix1 = max(0, min(w - 1, int(x1)))
    iy1 = max(0, min(h - 1, int(y1)))
    ix2 = max(ix1 + 1, min(w, int(x2)))
    iy2 = max(iy1 + 1, min(h, int(y2)))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _box_iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(1e-9, area_a + area_b - inter)
    return float(inter / union)


def _draw_single_box(frame, box, label_text: str, color=(255, 255, 255)):
    annotated = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        annotated,
        label_text,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def _save_snapshot(out_dir: str, video_stem: str, ts: float, best_conf: float, annotated) -> str:
    snap = f"{video_stem}_t{int(ts)}s_conf{int(best_conf * 100)}.jpg"
    cv2.imwrite(os.path.join(out_dir, snap), annotated)
    return snap


def _run_ultra_pass(
    model,
    frame,
    target: str,
    conf: float,
    iou: float,
    imgsz: Optional[int],
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    max_box_aspect_ratio: float,
) -> Optional[Dict[str, Any]]:
    predict_kwargs = {
        "source": frame,
        "conf": conf,
        "iou": iou,
        "verbose": False,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    results = model.predict(**predict_kwargs)
    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return None

    frame_h, frame_w = frame.shape[:2]

    best = None
    for b in boxes:
        cls_id = int(b.cls.item())
        cls_name = str(r.names.get(cls_id, "")).lower()
        score = float(b.conf.item())
        if cls_name != target:
            continue

        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        if not _passes_box_filters(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            frame_w=frame_w,
            frame_h=frame_h,
            min_box_area_ratio=min_box_area_ratio,
            max_box_area_ratio=max_box_area_ratio,
            max_box_aspect_ratio=max_box_aspect_ratio,
        ):
            continue

        if best is None or score > best["confidence"]:
            best = {
                "confidence": score,
                "box": (x1, y1, x2, y2),
                "class_name": cls_name,
            }

    if best is None:
        return None

    best["annotated"] = _draw_single_box(
        frame,
        best["box"],
        f"{best['class_name']} {best['confidence']:.2f}",
        color=(255, 255, 255),
    )
    return best


def _scan_one_video_pass(
    *,
    vp: Path,
    pass_name: str,
    yolo,
    out_dir: str,
    target: str,
    conf: float,
    iou: float,
    frame_skip: int,
    imgsz: Optional[int],
    save_every_hit: bool,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    max_box_aspect_ratio: float,
    edge_margin_ratio: float,
    blur_threshold: float,
    require_confirmation_for_suspicious: bool,
    confirm_window_sec: float,
    confirm_iou: float,
    cancel_event=None,
    LOG=None,
    progress_prefix: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {vp}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    stats: Dict[str, Any] = {
        "pass_name": pass_name,
        "conf": float(conf),
        "iou": float(iou),
        "frame_skip": int(frame_skip),
        "imgsz": imgsz,
        "frames_read": 0,
        "frames_inferred": 0,
        "video_elapsed_sec": None,
        "avg_infer_ms": None,
        "effective_sampled_fps": None,
        "cancelled": False,
        "candidates_seen": 0,
        "confirmed_hits": 0,
    }
    snapshots: List[Dict[str, Any]] = []
    candidate = None

    last_saved_sec = -1.0
    infer_time_total = 0.0
    t_video = time.perf_counter()

    frame_id = 0
    cancelled = False
    try:
        while True:
            if _is_cancelled(cancel_event):
                cancelled = True
                stats["cancelled"] = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            stats["frames_read"] += 1

            if frame_skip > 1 and (frame_id % frame_skip != 0):
                continue

            if _is_cancelled(cancel_event):
                cancelled = True
                stats["cancelled"] = True
                break

            ts = frame_id / fps
            stats["frames_inferred"] += 1

            t_inf0 = time.perf_counter()
            hit = _run_ultra_pass(
                model=yolo,
                frame=frame,
                target=target,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                min_box_area_ratio=min_box_area_ratio,
                max_box_area_ratio=max_box_area_ratio,
                max_box_aspect_ratio=max_box_aspect_ratio,
            )
            infer_time_total += (time.perf_counter() - t_inf0)

            if hit is None:
                continue

            x1, y1, x2, y2 = hit["box"]
            frame_h, frame_w = frame.shape[:2]
            edge_suspicious, edge_reasons = _edge_flags(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_w=frame_w,
                frame_h=frame_h,
                edge_margin_ratio=edge_margin_ratio,
            )
            blur_score = _blur_score_from_crop(frame, x1, y1, x2, y2)
            blur_suspicious = blur_score < float(blur_threshold)

            reasons = []
            if edge_suspicious:
                reasons.extend(edge_reasons)
            if blur_suspicious:
                reasons.append("blur")
            suspicious = require_confirmation_for_suspicious and len(reasons) > 0

            confirmed = True
            mode = "direct"
            if suspicious:
                confirmed = False
                stats["candidates_seen"] += 1
                if candidate is not None:
                    dt = ts - candidate["ts"]
                    overlap = _box_iou_xyxy(hit["box"], candidate["box"])
                    if dt <= float(confirm_window_sec) and overlap >= float(confirm_iou):
                        confirmed = True
                        mode = "confirmed"
                if not confirmed:
                    candidate = {
                        "ts": ts,
                        "box": hit["box"],
                        "confidence": hit["confidence"],
                        "reasons": reasons,
                        "blur_score": blur_score,
                    }
                    if LOG is not None:
                        LOG.info(
                            f"candidate video={vp.name} pass={pass_name} t={ts:.1f}s "
                            f"conf={hit['confidence']:.2f} reason={'+'.join(reasons)}"
                        )
                    continue

            if (not save_every_hit) and int(ts) == int(last_saved_sec):
                continue

            snap = _save_snapshot(
                out_dir=out_dir,
                video_stem=vp.stem,
                ts=ts,
                best_conf=hit["confidence"],
                annotated=hit["annotated"],
            )
            stats["confirmed_hits"] += 1

            snapshots.append(
                {
                    "timestamp_sec": float(ts),
                    "confidence": float(hit["confidence"]),
                    "snapshot": snap,
                    "scan_pass": pass_name,
                    "class_name": hit["class_name"],
                    "mode": mode,
                    "edge_suspicious": bool(edge_suspicious),
                    "blur_suspicious": bool(blur_suspicious),
                    "blur_score": float(blur_score),
                    "box": [float(v) for v in hit["box"]],
                }
            )
            last_saved_sec = ts
            candidate = None

            if LOG is not None:
                extra = "" if mode == "direct" else f" mode={mode}"
                LOG.info(
                    f"hit video={vp.name} pass={pass_name} t={ts:.1f}s "
                    f"class={hit['class_name']} conf={hit['confidence']:.2f}{extra}"
                )

            if not save_every_hit:
                break
    finally:
        cap.release()

    v_elapsed = time.perf_counter() - t_video
    stats["video_elapsed_sec"] = float(v_elapsed)

    if stats["frames_inferred"] > 0:
        stats["avg_infer_ms"] = float((infer_time_total / stats["frames_inferred"]) * 1000.0)
        stats["effective_sampled_fps"] = float(stats["frames_inferred"] / max(1e-9, v_elapsed))

    return stats, snapshots, cancelled


def _write_scan_report(report: Dict[str, Any], out_dir: str) -> str:
    report_path = os.path.join(out_dir, "scan_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    return report_path


def _is_ultralytics_detector_tag(tag: str) -> bool:
    t = (tag or "").strip().lower()
    return (
        t in ("yolo", "ultralytics", "rtdetr", "rtdetr-l", "rt-detr", "rt-detr-l")
        or t.startswith("yolo")
        or t.startswith("rtdetr")
        or t.startswith("rt-detr")
    )


def scan_videos_with_champion_detector(
    detector_run_dir: str | None = None,
    video_dir: str = "",
    output_run_dir: str = "",
    detector_path: str | None = None,
    detector_type: str = "auto",
    target_class: str = "red",
    conf: float = 0.5,
    iou: float = 0.5,
    frame_skip: int = 30,
    save_every_hit: bool = False,
    raw_dataset: str | None = None,
    imgsz: int | None = None,
    rescue_on_negative: bool = False,
    rescue_conf: float | None = None,
    rescue_frame_skip: int | None = None,
    rescue_imgsz: int | None = None,
    min_box_area_ratio: float = 0.0,
    max_box_area_ratio: float = 1.0,
    max_box_aspect_ratio: float = 10.0,
    edge_margin_ratio: float = 0.05,
    blur_threshold: float = 60.0,
    require_confirmation_for_suspicious: bool = True,
    confirm_window_sec: float = 1.5,
    confirm_iou: float = 0.30,
    cancel_event=None,
    LOG=None,
) -> Dict[str, Any]:
    if LOG is None:
        raise ValueError("LOG is required")

    vids = _list_videos(video_dir)
    if not vids:
        raise RuntimeError(f"No videos found in: {video_dir} (supported: {sorted(VIDEO_EXTS)})")

    champ = _resolve_detector_source(
        detector_run_dir=detector_run_dir,
        detector_path=detector_path,
        detector_type=detector_type,
    )

    tag = str(champ.get("tag", "")).lower()
    best_weights = champ.get("best_weights", None)
    if not best_weights or not os.path.exists(best_weights):
        raise FileNotFoundError(f"Detector weights not found: {best_weights}")

    out_dir = os.path.join(output_run_dir, "artifacts", "scan")
    _ensure_dir(out_dir)

    target = target_class.lower().strip()
    imgsz = _safe_int_or_none(imgsz)
    rescue_imgsz = _safe_int_or_none(rescue_imgsz)
    rescue_conf_eff = _resolve_rescue_conf(conf, rescue_conf) if rescue_on_negative else None
    rescue_frame_skip_eff = (
        _resolve_rescue_frame_skip(frame_skip, rescue_frame_skip) if rescue_on_negative else None
    )

    report: Dict[str, Any] = {
        "detector_run_dir": detector_run_dir,
        "detector_path": detector_path,
        "detector_type": _infer_detector_type(detector_path, detector_type) if detector_path else None,
        "output_run_dir": output_run_dir,
        "video_dir": video_dir,
        "target_class": target,
        "conf": float(conf),
        "iou": float(iou),
        "frame_skip": int(frame_skip),
        "imgsz": imgsz,
        "save_every_hit": bool(save_every_hit),
        "rescue_on_negative": bool(rescue_on_negative),
        "rescue_conf": rescue_conf_eff,
        "rescue_frame_skip": rescue_frame_skip_eff,
        "rescue_imgsz": rescue_imgsz if rescue_imgsz is not None else imgsz,
        "box_filters": {
            "min_box_area_ratio": float(min_box_area_ratio),
            "max_box_area_ratio": float(max_box_area_ratio),
            "max_box_aspect_ratio": float(max_box_aspect_ratio),
        },
        "suspicious_hit_filters": {
            "edge_margin_ratio": float(edge_margin_ratio),
            "blur_threshold": float(blur_threshold),
            "require_confirmation_for_suspicious": bool(require_confirmation_for_suspicious),
            "confirm_window_sec": float(confirm_window_sec),
            "confirm_iou": float(confirm_iou),
        },
        "champion": champ,
        "detector": {
            "source_mode": champ.get("source_mode"),
            "tag": champ.get("tag"),
            "best_weights": champ.get("best_weights"),
            "display_name": champ.get("display_name"),
        },
        "videos_found": len(vids),
        "videos_with_target": 0,
        "total_snapshots": 0,
        "videos": [],
        "runtime_seconds": None,
        "cancelled": False,
        "cancelled_at_video": None,
        "cancelled_at_pass": None,
        "videos_with_detections": [],
    }

    source_desc = (
        f"detector_file={Path(best_weights).name} detector_type={champ.get('detector_type', tag)}"
        if champ.get("source_mode") == "detector_path"
        else f"detector_run_dir={detector_run_dir}"
    )
    LOG.info(
        f"[START] component=Scan videos_found={len(vids)} target_class={target} "
        f"conf={conf:.2f} frame_skip={frame_skip} imgsz={imgsz if imgsz is not None else 'default'} "
        f"{source_desc}"
    )

    if not _is_ultralytics_detector_tag(tag):
        raise ValueError(f"Unsupported champion tag for scanning: {tag}")

    yolo = _load_ultralytics_model(best_weights)
    LOG.info(f"load_model backend=ultralytics detector_tag={tag} weights={Path(best_weights).name}")

    t0 = time.perf_counter()

    for idx, vp in enumerate(vids, start=1):
        if _is_cancelled(cancel_event):
            report["cancelled"] = True
            report["cancelled_at_video"] = vp.name
            report["cancelled_at_pass"] = "before_video"
            LOG.warning(f"cancelled video={vp.name} pass=before_video videos_completed={len(report['videos'])}")
            break

        video_t0 = time.perf_counter()
        LOG.info(f"({idx}/{len(vids)}) video={vp.name}")

        ventry: Dict[str, Any] = {
            "video": vp.name,
            "video_path": str(vp),
            "fps": None,
            "scan_pass": "none",
            "frames_read": 0,
            "frames_inferred": 0,
            "snapshots": [],
            "video_elapsed_sec": None,
            "avg_infer_ms": None,
            "effective_sampled_fps": None,
            "primary": None,
            "rescue": None,
        }

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            cap.release()
            LOG.warning(f"({idx}/{len(vids)}) done video={vp.name} snapshots=0 elapsed_sec=0.0 reason=open_failed")
            report["videos"].append(ventry)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps <= 0:
            fps = 30.0
        ventry["fps"] = float(fps)

        primary_stats, primary_snapshots, primary_cancelled = _scan_one_video_pass(
            vp=vp,
            pass_name="primary",
            yolo=yolo,
            out_dir=out_dir,
            target=target,
            conf=conf,
            iou=iou,
            frame_skip=frame_skip,
            imgsz=imgsz,
            save_every_hit=save_every_hit,
            min_box_area_ratio=min_box_area_ratio,
            max_box_area_ratio=max_box_area_ratio,
            max_box_aspect_ratio=max_box_aspect_ratio,
            edge_margin_ratio=edge_margin_ratio,
            blur_threshold=blur_threshold,
            require_confirmation_for_suspicious=require_confirmation_for_suspicious,
            confirm_window_sec=confirm_window_sec,
            confirm_iou=confirm_iou,
            cancel_event=cancel_event,
            LOG=LOG,
            progress_prefix=f"({idx}/{len(vids)})",
        )

        ventry["primary"] = primary_stats
        ventry["frames_read"] += int(primary_stats["frames_read"])
        ventry["frames_inferred"] += int(primary_stats["frames_inferred"])
        ventry["video_elapsed_sec"] = float(primary_stats["video_elapsed_sec"] or 0.0)
        if primary_stats["avg_infer_ms"] is not None:
            ventry["avg_infer_ms"] = float(primary_stats["avg_infer_ms"])
        if primary_stats["effective_sampled_fps"] is not None:
            ventry["effective_sampled_fps"] = float(primary_stats["effective_sampled_fps"])

        snapshots = list(primary_snapshots)
        scan_pass = "primary" if snapshots else "none"

        if primary_cancelled:
            report["cancelled"] = True
            report["cancelled_at_video"] = vp.name
            report["cancelled_at_pass"] = "primary"

        if (not snapshots) and rescue_on_negative and (not report["cancelled"]):
            rescue_stats, rescue_snapshots, rescue_cancelled = _scan_one_video_pass(
                vp=vp,
                pass_name="rescue",
                yolo=yolo,
                out_dir=out_dir,
                target=target,
                conf=float(rescue_conf_eff),
                iou=iou,
                frame_skip=int(rescue_frame_skip_eff),
                imgsz=rescue_imgsz if rescue_imgsz is not None else imgsz,
                save_every_hit=save_every_hit,
                min_box_area_ratio=min_box_area_ratio,
                max_box_area_ratio=max_box_area_ratio,
                max_box_aspect_ratio=max_box_aspect_ratio,
                edge_margin_ratio=edge_margin_ratio,
                blur_threshold=blur_threshold,
                require_confirmation_for_suspicious=require_confirmation_for_suspicious,
                confirm_window_sec=confirm_window_sec,
                confirm_iou=confirm_iou,
                cancel_event=cancel_event,
                LOG=LOG,
                progress_prefix=f"({idx}/{len(vids)})",
            )
            ventry["rescue"] = rescue_stats
            ventry["frames_read"] += int(rescue_stats["frames_read"])
            ventry["frames_inferred"] += int(rescue_stats["frames_inferred"])
            ventry["video_elapsed_sec"] += float(rescue_stats["video_elapsed_sec"] or 0.0)

            total_ms = 0.0
            total_frames = 0
            for stats in (primary_stats, rescue_stats):
                if stats["avg_infer_ms"] is not None and stats["frames_inferred"] > 0:
                    total_ms += float(stats["avg_infer_ms"]) * int(stats["frames_inferred"])
                    total_frames += int(stats["frames_inferred"])
            if total_frames > 0:
                ventry["avg_infer_ms"] = total_ms / total_frames
            if float(ventry["video_elapsed_sec"]) > 0.0:
                ventry["effective_sampled_fps"] = (
                    float(ventry["frames_inferred"]) / max(1e-9, float(ventry["video_elapsed_sec"]))
                )

            if rescue_snapshots:
                snapshots = rescue_snapshots
                scan_pass = "rescue"
            if rescue_cancelled:
                report["cancelled"] = True
                report["cancelled_at_video"] = vp.name
                report["cancelled_at_pass"] = "rescue"

        ventry["scan_pass"] = scan_pass
        ventry["snapshots"] = snapshots
        report["videos"].append(ventry)

        if snapshots:
            report["videos_with_target"] += 1
            report["total_snapshots"] += len(snapshots)
            report["videos_with_detections"].append(vp.name)

        elapsed = time.perf_counter() - video_t0
        LOG.info(
            f"({idx}/{len(vids)}) done video={vp.name} pass={scan_pass} snapshots={len(snapshots)} elapsed_sec={elapsed:.1f}"
        )

        if report["cancelled"]:
            break

    report["runtime_seconds"] = float(time.perf_counter() - t0)
    report_path = _write_scan_report(report, out_dir)

    LOG.info(f"[ARTIFACT] component=Scan scan_report={report_path}")
    LOG.info(
        f"[RESULT] component=Scan videos_found={report['videos_found']} "
        f"videos_completed={len(report['videos'])} videos_with_target={report['videos_with_target']} "
        f"total_snapshots={report['total_snapshots']} runtime_seconds={report['runtime_seconds']:.1f} "
        f"cancelled={report['cancelled']}"
    )

    if report["videos_with_detections"]:
        LOG.info("Videos with squirrel detections:")
        for name in report["videos_with_detections"]:
            LOG.info(f" - {name}")

    LOG.info(f"[DONE] component=Scan elapsed_sec={report['runtime_seconds']:.1f}")
    return report
