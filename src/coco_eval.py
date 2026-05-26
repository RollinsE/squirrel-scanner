import numpy as np
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _to_xywh(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def evaluate_coco_map(model, dataloader, device, score_thresh: float = 0.05, LOG=None, model_name: str = "Detector"):
    """
    Compute COCO-style mAP for torchvision detectors.

    Returns dict:
      map50_95, map50, map75, mar1, mar10, mar100
    """
    model.eval()

    coco_gt = COCO()
    coco_gt.dataset = {"images": [], "annotations": [], "categories": []}

    ann_id = 1
    max_cat = 0
    images_meta = {}

    for images, targets in dataloader:
        for img, tgt in zip(images, targets):
            img_id = int(tgt["image_id"].item())
            _, h, w = img.shape
            images_meta[img_id] = (w, h)

            labels = tgt["labels"]
            if isinstance(labels, torch.Tensor) and labels.numel() > 0:
                max_cat = max(max_cat, int(labels.max().item()))

    coco_gt.dataset["categories"] = [{"id": i, "name": str(i)} for i in range(1, max_cat + 1)]

    gt_ann_count = 0
    for images, targets in dataloader:
        for img, tgt in zip(images, targets):
            img_id = int(tgt["image_id"].item())
            w, h = images_meta[img_id]

            coco_gt.dataset["images"].append({
                "id": img_id,
                "width": int(w),
                "height": int(h),
                "file_name": str(img_id),
            })

            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]
            gt_areas = tgt.get("area", torch.zeros((gt_boxes.shape[0],), dtype=torch.float32))
            gt_iscrowd = tgt.get("iscrowd", torch.zeros((gt_boxes.shape[0],), dtype=torch.int64))

            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.numpy()
            if isinstance(gt_areas, torch.Tensor):
                gt_areas = gt_areas.numpy()
            if isinstance(gt_iscrowd, torch.Tensor):
                gt_iscrowd = gt_iscrowd.numpy()

            for b, lab, area, crowd in zip(gt_boxes, gt_labels, gt_areas, gt_iscrowd):
                coco_gt.dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(lab),
                    "bbox": _to_xywh(b),
                    "area": float(area),
                    "iscrowd": int(crowd),
                })
                ann_id += 1
                gt_ann_count += 1

    coco_gt.createIndex()

    coco_results = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                img_id = int(tgt["image_id"].item())
                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()

                keep = scores >= score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                for b, s, lab in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": int(lab),
                        "bbox": _to_xywh(b),
                        "score": float(s),
                    })

    if LOG is not None:
        LOG.info(
            f"[EVAL] model={model_name} coco_gt_images={len(coco_gt.dataset['images'])} "
            f"coco_gt_annotations={gt_ann_count} predictions={len(coco_results)} score_thresh={score_thresh}"
        )

    if len(coco_results) == 0:
        return {"map50_95": 0.0, "map50": 0.0, "map75": 0.0, "mar1": 0.0, "mar10": 0.0, "mar100": 0.0}

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "map50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "mar1": float(stats[6]),
        "mar10": float(stats[7]),
        "mar100": float(stats[8]),
    }
