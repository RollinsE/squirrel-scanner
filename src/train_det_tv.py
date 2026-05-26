import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import time
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.amp import autocast, GradScaler

from src.detect_dataset import YoloDetectionDataset, collate_fn
from src.coco_eval import evaluate_coco_map
from src.logger import log_stage, component_stage


def _device(device_arg: str | None):
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = str(device_arg).lower().strip()
    if d == "cpu":
        return torch.device("cpu")
    if d in ("cuda", "cuda:0"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_setup(dev: torch.device):
    use_amp = (dev.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    return use_amp, scaler


def _num_classes_from_yaml(raw_dataset: str) -> int:
    import yaml
    ypath = os.path.join(raw_dataset, "data.yaml")
    with open(ypath, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d.get("names", [])
    return len(names)


def _make_loaders(raw_dataset: str, batch: int, requested_workers: int, LOG):
    safe_workers = 0
    if requested_workers != safe_workers:
        log_stage(
            LOG, "WARN", "Torchvision",
            reason="forcing_num_workers_0_for_colab_stability",
            requested=requested_workers, used=safe_workers
        )

    train_ds = YoloDetectionDataset(raw_dataset, "train", transforms=ToTensor())
    val_ds = YoloDetectionDataset(raw_dataset, "valid", transforms=ToTensor())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=safe_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=safe_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def _plot_history(history: list[dict], out_path: str, title: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_epochs = [h["epoch"] for h in history if h["val_map50_95"] is not None]
    val_map = [h["val_map50_95"] for h in history if h["val_map50_95"] is not None]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    if val_epochs:
        plt.plot(val_epochs, val_map, label="val_mAP50-95")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _checkpoint_paths(run_dir: str):
    out_dir = os.path.join(run_dir, "artifacts", "detectors", "fasterrcnn")
    os.makedirs(out_dir, exist_ok=True)
    return {
        "dir": out_dir,
        "ckpt": os.path.join(out_dir, "checkpoint.pt"),
        "best": os.path.join(out_dir, "best.pth"),
        "hist": os.path.join(out_dir, "history.json"),
    }


def _save_checkpoint(path: str, epoch: int, best_map: float, model, optim, scaler, history: list[dict]):
    torch.save({
        "epoch": epoch,
        "best_map": best_map,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "history": history,
    }, path)


def _load_checkpoint(path: str, model, optim, scaler, LOG):
    """Load a detector checkpoint and restore training state when available.

    Returns: (start_epoch, best_map, history)
    """
    ckpt = torch.load(path, map_location="cpu")
    start_epoch = int(ckpt.get("epoch", 0))
    best_map = float(ckpt.get("best_map", -1.0))
    history = ckpt.get("history", [])

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])

    if "optim" in ckpt and ckpt["optim"] is not None:
        try:
            optim.load_state_dict(ckpt["optim"])
        except Exception as e:
            log_stage(LOG, "WARN", "FasterRCNN", reason="optimizer_state_load_failed", error=str(e))
    else:
        log_stage(LOG, "WARN", "FasterRCNN", reason="checkpoint_missing_optim_state", action="continue_with_fresh_optimizer")

    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            log_stage(LOG, "WARN", "FasterRCNN", reason="scaler_state_load_failed", error=str(e))

    return start_epoch, best_map, history


def _finite_loss_or_skip(losses: torch.Tensor, optim, LOG, epoch: int, epochs: int, i: int, num_batches: int) -> bool:
    if torch.isfinite(losses):
        return True
    try:
        loss_val = float(losses.detach().cpu())
    except Exception:
        loss_val = None
    log_stage(
        LOG, "WARN", "FasterRCNN",
        epoch=f"{epoch}/{epochs}",
        batch=f"{i+1}/{num_batches}",
        reason="non_finite_loss",
        loss=loss_val
    )
    optim.zero_grad(set_to_none=True)
    return False


def train_fasterrcnn(
    raw_dataset: str,
    run_dir: str,
    epochs: int,
    batch: int,
    lr: float,
    num_workers: int,
    device_arg: str | None,
    eval_every: int,
    resume: bool,
    LOG
) -> dict:
    dev = _device(device_arg)
    use_amp, scaler = _amp_setup(dev)
    num_classes = _num_classes_from_yaml(raw_dataset) + 1  # + background

    paths = _checkpoint_paths(run_dir)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    train_loader, val_loader, n_train, n_val = _make_loaders(raw_dataset, batch, num_workers, LOG)

    with component_stage(LOG, "FasterRCNN", epochs=epochs, batch=batch, lr=lr, device=str(dev), eval_every=eval_every, amp=use_amp):
        log_stage(LOG, "INFO", "FasterRCNN", train_images=n_train, val_images=n_val, num_classes_with_bg=num_classes)

        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.to(dev)

        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=lr)

        best_map = -1.0
        history = []
        start_epoch = 0

        # Resume checkpoint (if present)
        if resume and os.path.isfile(paths["ckpt"]):
            log_stage(LOG, "INFO", "FasterRCNN", message="Checkpoint discovery", checkpoint=paths["ckpt"], best=paths["best"], resume_requested=True)
            start_epoch, best_map, history = _load_checkpoint(paths["ckpt"], model, optim, scaler, LOG)

            if start_epoch >= epochs:
                log_stage(LOG, "DONE", "FasterRCNN", message="Checkpoint already completed requested epochs; skipping.", completed_epochs=start_epoch, requested_epochs=epochs, best=paths["best"], best_map=best_map)
                return {"tag": "fasterrcnn", "best_weights": paths["best"], "score": float(best_map), "skipped": True}

        for epoch in range(start_epoch + 1, epochs + 1):
            model.train()
            t_epoch = time.time()

            t_first = time.time()
            first_batch_seen = False

            loss_sum = 0.0
            num_batches = len(train_loader)
            bad_batches = 0

            for i, (images, targets) in enumerate(train_loader):
                if not first_batch_seen:
                    first_batch_seen = True
                    log_stage(
                        LOG, "INFO", "FasterRCNN",
                        epoch=f"{epoch}/{epochs}",
                        first_batch_after_sec=f"{time.time()-t_first:.1f}",
                        batches=num_batches
                    )

                images = [img.to(dev) for img in images]
                targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]

                with autocast("cuda", enabled=use_amp):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                if not _finite_loss_or_skip(losses, optim, LOG, epoch, epochs, i, num_batches):
                    bad_batches += 1
                    continue

                optim.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(losses).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optim.step()

                loss_sum += float(losses.item())

                if (i + 1) % 100 == 0 or (i + 1) == num_batches:
                    denom = max(1, (i + 1) - bad_batches)
                    log_stage(
                        LOG, "INFO", "FasterRCNN",
                        epoch=f"{epoch}/{epochs}",
                        batch=f"{i+1}/{num_batches}",
                        running_loss=f"{loss_sum/denom:.4f}",
                        bad_batches=bad_batches
                    )

            denom = max(1, num_batches - bad_batches)
            train_loss = loss_sum / denom

            log_stage(
                LOG, "EPOCH", "FasterRCNN",
                epoch=f"{epoch}/{epochs}",
                train_loss=f"{train_loss:.4f}",
                bad_batches=bad_batches,
                epoch_sec=f"{time.time()-t_epoch:.1f}"
            )

            # evaluation schedule
            val_map = None
            val_map50 = None
            if eval_every <= 1 or (epoch % eval_every == 0) or (epoch == epochs):
                coco = evaluate_coco_map(model, val_loader, dev, score_thresh=0.05, LOG=LOG, model_name="FasterRCNN")
                val_map = float(coco["map50_95"])
                val_map50 = float(coco["map50"])
                log_stage(LOG, "EVAL", "FasterRCNN", epoch=f"{epoch}/{epochs}", map50_95=f"{val_map:.4f}", map50=f"{val_map50:.4f}")

                if val_map > best_map:
                    prev = best_map
                    best_map = val_map
                    torch.save(model.state_dict(), paths["best"])
                    log_stage(LOG, "CHECKPOINT", "FasterRCNN", epoch=f"{epoch}/{epochs}", improved_from=f"{prev:.4f}", improved_to=f"{best_map:.4f}", path=paths["best"])
            else:
                log_stage(LOG, "EVAL", "FasterRCNN", epoch=f"{epoch}/{epochs}", skipped=True)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_map50_95": val_map, "val_map50": val_map50, "bad_batches": bad_batches})

            # always save resume checkpoint each epoch
            _save_checkpoint(paths["ckpt"], epoch, best_map, model, optim, scaler, history)
            log_stage(LOG, "INFO", "FasterRCNN", checkpoint_saved=paths["ckpt"], epoch=epoch)

        # save history + plot
        with open(paths["hist"], "w", encoding="utf-8") as f:
            json.dump({"best_map50_95": best_map, "history": history}, f, indent=4)

        plot_path = os.path.join(plots_dir, "fasterrcnn_loss_vs_map.png")
        _plot_history(history, plot_path, "Faster R-CNN: Train Loss vs Val mAP50-95")

        log_stage(LOG, "ARTIFACT", "FasterRCNN", history=paths["hist"], plot=plot_path)
        log_stage(LOG, "DONE", "FasterRCNN", best=paths["best"], score=best_map)

        return {"tag": "fasterrcnn", "best_weights": paths["best"], "score": float(best_map)}
