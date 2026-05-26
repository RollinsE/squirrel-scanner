import argparse
import os
import sys

os.environ["WANDB_DISABLED"] = "true"
sys.modules["wandb"] = None

import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
if hasattr(torch.serialization, "load"):
    torch.serialization.load = _patched_torch_load

from src.experiment import create_run_dir, find_latest_run_dir
from src.logger import setup_logger, log_stage, component_stage
from src.acquisition import acquire_dataset_from_roboflow
from src.dataset_merge import merge_yolo_detection_datasets, parse_class_map, parse_csv_list
from src.preprocess import convert_yolo_detection_to_imagefolder
from src.train_det import parse_model_batches, parse_model_list, train_detectors
from src.scan_det_any import scan_videos_with_champion_detector


def parse_args():
    p = argparse.ArgumentParser("Squirrel Scanner Pipeline")

    p.add_argument("--mode", required=True, choices=["acquire", "acquire_multi", "merge_datasets", "train_det", "scan_det"])
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    p.add_argument(
        "--run_base_dir",
        default="runs",
        help="Base folder for timestamped runs, logs, artifacts, metrics, plots, and scan outputs.",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="Use a specific existing run directory, e.g. runs/run_YYYYMMDD_HHMMSS.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training inside the chosen run directory (either --run_dir, or latest under --run_base_dir).",
    )

    p.add_argument("--workspace", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--rf_format", default="yolov8")
    p.add_argument("--download_dir", default=None)
    p.add_argument(
        "--roboflow_datasets",
        nargs="*",
        default=None,
        help=(
            "For acquire_multi: one or more Roboflow dataset specs in "
            "workspace/project/version or workspace:project:version format."
        ),
    )

    p.add_argument("--raw_dataset", default=None)
    p.add_argument(
        "--raw_datasets",
        nargs="*",
        default=None,
        help="One or more existing YOLO dataset roots to merge/train from.",
    )
    p.add_argument(
        "--merged_dataset_dir",
        default=None,
        help="Output folder for merged YOLO dataset. Defaults to run_dir/artifacts/merged_dataset when training.",
    )
    p.add_argument(
        "--target_classes",
        default=None,
        help="Comma-separated canonical classes for merged datasets, e.g. red,grey,marten,rat.",
    )
    p.add_argument(
        "--class_map",
        default=None,
        help=(
            "Optional class alias mapping for merging, e.g. "
            "'red squirrels=red,grey squirrels=grey,rats=rat'."
        ),
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--models",
        default="yolo11n",
        help=(
            "Comma-separated Ultralytics model names/weights to train, in order. "
            "Examples: yolo11n or yolo11n,yolo11s,rtdetr-l."
        ),
    )
    p.add_argument("--batch", type=int, default=8, help="Default batch size for models without a specific batch option.")
    p.add_argument("--batch_yolo11", type=int, default=8)
    p.add_argument("--batch_yolo26", type=int, default=8)
    p.add_argument("--batch_rtdetr", type=int, default=8)
    p.add_argument(
        "--model_batches",
        default=None,
        help="Optional per-model batches, e.g. yolo11n=8,yolo11s=4,rtdetr-l=2.",
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate passed to Ultralytics as lr0.")
    p.add_argument("--optimizer", default="auto", help="Ultralytics optimizer, e.g. auto, SGD, Adam, AdamW, NAdam, RMSProp.")
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1,...")

    p.add_argument("--detector_run_dir", default=None)
    p.add_argument("--video_dir", default=None)
    p.add_argument("--target_class", default="red")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--frame_skip", type=int, default=30)
    p.add_argument("--save_every_hit", action="store_true")
    p.add_argument("--scan_imgsz", type=int, default=640)
    p.add_argument("--rescue_on_negative", action="store_true")
    p.add_argument("--rescue_conf", type=float, default=None)
    p.add_argument("--rescue_frame_skip", type=int, default=None)
    p.add_argument("--rescue_imgsz", type=int, default=None)
    p.add_argument("--min_box_area_ratio", type=float, default=0.0)
    p.add_argument("--max_box_area_ratio", type=float, default=1.0)
    p.add_argument("--max_box_aspect_ratio", type=float, default=10.0)

    return p.parse_args()


def resolve_device(device_arg: str, LOG) -> str:
    d = (device_arg or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()

    if d == "auto":
        resolved = "0" if cuda_available else "cpu"
        log_stage(LOG, "INFO", "Pipeline", action="resolve_device", requested=device_arg, ultra_device=resolved)
        return resolved

    if d == "cpu":
        log_stage(LOG, "INFO", "Pipeline", action="resolve_device", requested=device_arg, ultra_device="cpu")
        return "cpu"

    if d in ("cuda", "cuda:0"):
        if not cuda_available:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        log_stage(LOG, "INFO", "Pipeline", action="resolve_device", requested=device_arg, ultra_device="0")
        return "0"

    if all(part.strip().isdigit() for part in d.split(",")):
        if not cuda_available:
            raise RuntimeError("CUDA indices requested but torch.cuda.is_available() is False.")
        log_stage(LOG, "INFO", "Pipeline", action="resolve_device", requested=device_arg, ultra_device=d)
        return d

    raise ValueError(f"Invalid --device '{device_arg}'. Use auto, cpu, 0, or 0,1,...")


def _derive_run_id(run_dir: str) -> str:
    base = os.path.basename(run_dir.rstrip("/"))
    if base.startswith("run_"):
        return base.replace("run_", "")
    return base


def _parse_roboflow_dataset_spec(spec: str) -> tuple[str, str, int]:
    value = (spec or "").strip()
    if not value:
        raise ValueError("Empty Roboflow dataset spec.")

    if ":" in value and value.count(":") == 2:
        workspace, project, version = value.split(":")
    else:
        parts = value.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(
                "Roboflow dataset specs must be workspace/project/version "
                "or workspace:project:version."
            )
        workspace, project, version = parts

    return workspace.strip(), project.strip(), int(version)


def _resolve_dataset_for_training(args, run_dir: str, LOG) -> str:
    roots = []
    if args.raw_dataset:
        roots.append(args.raw_dataset)
    if args.raw_datasets:
        roots.extend(args.raw_datasets)

    # Preserve order while removing accidental duplicates.
    deduped = []
    seen = set()
    for root in roots:
        key = os.path.abspath(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)

    if not deduped:
        raise ValueError("Training requires --raw_dataset or --raw_datasets")

    if len(deduped) == 1:
        return deduped[0]

    merged_dir = args.merged_dataset_dir or os.path.join(run_dir, "artifacts", "merged_dataset")
    return merge_yolo_detection_datasets(
        dataset_roots=deduped,
        output_dir=merged_dir,
        target_classes=parse_csv_list(args.target_classes),
        class_map=parse_class_map(args.class_map),
        run_dir=run_dir,
        LOG=LOG,
    )


def main():
    args = parse_args()

    if args.run_dir:
        run_dir = args.run_dir
        run_id = _derive_run_id(run_dir)
        os.makedirs(run_dir, exist_ok=True)
    else:
        if args.resume:
            latest = find_latest_run_dir(args.run_base_dir)
            if not latest:
                print(f"[FATAL] --resume set but no run_ folders under {args.run_base_dir}")
                sys.exit(1)
            run_dir = latest
            run_id = _derive_run_id(run_dir)
        else:
            run_id, run_dir = create_run_dir(base=args.run_base_dir)

    LOG = setup_logger(run_id, run_dir, log_level=args.log_level)

    try:
        with component_stage(
            LOG,
            "Pipeline",
            mode=args.mode,
            run_id=run_id,
            run_dir=run_dir,
            resume=args.resume,
        ):
            if args.mode == "acquire":
                api_key = os.getenv("ROBOFLOW_API_KEY")
                if not api_key:
                    raise ValueError("ROBOFLOW_API_KEY env var is not set.")

                if not (args.workspace and args.project and args.version and args.download_dir):
                    raise ValueError("Acquire requires --workspace --project --version --download_dir")

                raw_root = acquire_dataset_from_roboflow(
                    api_key=api_key,
                    workspace=args.workspace,
                    project=args.project,
                    version=args.version,
                    rf_format=args.rf_format,
                    download_dir=args.download_dir,
                    run_dir=run_dir,
                    LOG=LOG,
                )

                cls_root = convert_yolo_detection_to_imagefolder(raw_root, LOG=LOG)
                log_stage(LOG, "RESULT", "Pipeline", mode="acquire", raw_dataset=raw_root, classification_dataset=cls_root)

            elif args.mode == "acquire_multi":
                api_key = os.getenv("ROBOFLOW_API_KEY")
                if not api_key:
                    raise ValueError("ROBOFLOW_API_KEY env var is not set.")
                if not args.roboflow_datasets:
                    raise ValueError("acquire_multi requires --roboflow_datasets")
                if not args.download_dir:
                    raise ValueError("acquire_multi requires --download_dir")

                downloaded_roots = []
                for spec in args.roboflow_datasets:
                    workspace, project, version = _parse_roboflow_dataset_spec(spec)
                    downloaded_roots.append(
                        acquire_dataset_from_roboflow(
                            api_key=api_key,
                            workspace=workspace,
                            project=project,
                            version=version,
                            rf_format=args.rf_format,
                            download_dir=args.download_dir,
                            run_dir=run_dir,
                            LOG=LOG,
                        )
                    )

                merged_dir = args.merged_dataset_dir or os.path.join(args.download_dir, "merged_squirrel_dataset")
                merged_root = merge_yolo_detection_datasets(
                    dataset_roots=downloaded_roots,
                    output_dir=merged_dir,
                    target_classes=parse_csv_list(args.target_classes),
                    class_map=parse_class_map(args.class_map),
                    run_dir=run_dir,
                    LOG=LOG,
                )
                log_stage(LOG, "RESULT", "Pipeline", mode="acquire_multi", merged_dataset=merged_root)

            elif args.mode == "merge_datasets":
                roots = []
                if args.raw_dataset:
                    roots.append(args.raw_dataset)
                if args.raw_datasets:
                    roots.extend(args.raw_datasets)
                if not roots:
                    raise ValueError("merge_datasets requires --raw_dataset and/or --raw_datasets")
                if not args.merged_dataset_dir:
                    raise ValueError("merge_datasets requires --merged_dataset_dir")

                merged_root = merge_yolo_detection_datasets(
                    dataset_roots=roots,
                    output_dir=args.merged_dataset_dir,
                    target_classes=parse_csv_list(args.target_classes),
                    class_map=parse_class_map(args.class_map),
                    run_dir=run_dir,
                    LOG=LOG,
                )
                log_stage(LOG, "RESULT", "Pipeline", mode="merge_datasets", merged_dataset=merged_root)

            elif args.mode == "train_det":
                raw_dataset = _resolve_dataset_for_training(args, run_dir, LOG)

                ultra_device = resolve_device(args.device, LOG)
                log_stage(
                    LOG,
                    "INFO",
                    "Pipeline",
                    mode="train_det",
                    raw_dataset=raw_dataset,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    models=args.models,
                    batch=args.batch,
                    batch_yolo11=args.batch_yolo11,
                    batch_yolo26=args.batch_yolo26,
                    batch_rtdetr=args.batch_rtdetr,
                    model_batches=args.model_batches,
                    eval_every=args.eval_every,
                    lr=args.lr,
                    optimizer=args.optimizer,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                )

                train_detectors(
                    raw_dataset=raw_dataset,
                    run_dir=run_dir,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch_yolo11=args.batch_yolo11,
                    batch_yolo26=args.batch_yolo26,
                    batch_rtdetr=args.batch_rtdetr,
                    lr=args.lr,
                    num_workers=args.num_workers,
                    ultra_device=ultra_device,
                    eval_every=args.eval_every,
                    resume=args.resume,
                    LOG=LOG,
                    models=parse_model_list(args.models),
                    model_batches=parse_model_batches(args.model_batches),
                    default_batch=args.batch,
                    optimizer=args.optimizer,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                )
                log_stage(LOG, "RESULT", "Pipeline", mode=args.mode, status="success")

            elif args.mode == "scan_det":
                if not args.detector_run_dir:
                    raise ValueError("--detector_run_dir is required for scan_det")
                if not args.video_dir:
                    raise ValueError("--video_dir is required for scan_det")

                report = scan_videos_with_champion_detector(
                    detector_run_dir=args.detector_run_dir,
                    video_dir=args.video_dir,
                    output_run_dir=run_dir,
                    target_class=args.target_class,
                    conf=args.conf,
                    iou=args.iou,
                    frame_skip=args.frame_skip,
                    save_every_hit=args.save_every_hit,
                    raw_dataset=args.raw_dataset,
                    imgsz=args.scan_imgsz,
                    rescue_on_negative=args.rescue_on_negative,
                    rescue_conf=args.rescue_conf,
                    rescue_frame_skip=args.rescue_frame_skip,
                    rescue_imgsz=args.rescue_imgsz,
                    min_box_area_ratio=args.min_box_area_ratio,
                    max_box_area_ratio=args.max_box_area_ratio,
                    max_box_aspect_ratio=args.max_box_aspect_ratio,
                    LOG=LOG,
                )
                log_stage(
                    LOG,
                    "RESULT",
                    "Pipeline",
                    mode="scan_det",
                    videos_found=report.get("videos_found"),
                    videos_with_target=report.get("videos_with_target"),
                    total_snapshots=report.get("total_snapshots"),
                )

            log_stage(LOG, "DONE", "Pipeline", mode=args.mode, status="success")

    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
