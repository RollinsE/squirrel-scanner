import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _find_images(images_dir: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    return sorted(files)


class YoloDetectionDataset(Dataset):
    """
    Torchvision-style detection dataset reading YOLO txt labels.

    Critical detail:
      - For images with NO objects, torchvision still requires:
          boxes: Tensor[0,4] (NOT Tensor[0])
          labels: Tensor[0]
          area: Tensor[0]
          iscrowd: Tensor[0]
    """

    def __init__(self, root: str, split: str, transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms

        self.images_dir = os.path.join(root, split, "images")
        self.labels_dir = os.path.join(root, split, "labels")

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(self.images_dir)
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(self.labels_dir)

        self.image_files = _find_images(self.images_dir)
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, f"{stem}.txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes_list = []
        labels_list = []
        areas_list = []
        iscrowd_list = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue

                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])

                # YOLO normalized -> pixel xyxy
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h

                # clamp
                x1 = max(0.0, min(x1, w - 1.0))
                y1 = max(0.0, min(y1, h - 1.0))
                x2 = max(0.0, min(x2, w - 1.0))
                y2 = max(0.0, min(y2, h - 1.0))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes_list.append([x1, y1, x2, y2])
                labels_list.append(cls + 1)  # background=0, classes start at 1

                areas_list.append((x2 - x1) * (y2 - y1))
                iscrowd_list.append(0)

        # always return correct shapes
        if len(boxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
            areas = torch.tensor(areas_list, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd_list, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
