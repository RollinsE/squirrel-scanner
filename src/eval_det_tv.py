"""Simple validation utilities for Torchvision detection models."""

import torch


def _box_iou(boxes1, boxes2):
    """
    boxes: [N,4] in xyxy
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.max(x11[:, None], x21[None, :])
    inter_y1 = torch.max(y11[:, None], y21[None, :])
    inter_x2 = torch.min(x12[:, None], x22[None, :])
    inter_y2 = torch.min(y12[:, None], y22[None, :])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()


def evaluate_detector_simple(model, val_loader, device, score_thresh: float = 0.3) -> float:
    """
    Returns a simple validation score:
      mean of best IoU per GT box, across the validation set,
      using predictions above score_thresh.
    Higher is better.
    """
    model.eval()
    total = 0.0
    count = 0

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_boxes = tgt["boxes"].to(device)
            if gt_boxes.numel() == 0:
                continue

            pred_boxes = out["boxes"].to(device)
            pred_scores = out["scores"].to(device)

            keep = pred_scores >= score_thresh
            pred_boxes = pred_boxes[keep]

            if pred_boxes.numel() == 0:
                # no predictions: contributes 0
                total += 0.0
                count += int(gt_boxes.shape[0])
                continue

            ious = _box_iou(gt_boxes, pred_boxes)  # [G, P]
            best_per_gt = ious.max(dim=1).values  # [G]
            total += float(best_per_gt.sum().item())
            count += int(best_per_gt.numel())

    return total / max(1, count)
