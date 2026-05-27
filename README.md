# Squirrel Scanner

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-<LICENSE>-lightgrey.svg)](LICENSE)

Squirrel Scanner was created to support red squirrel conservation work on Brownsea Island.

During routine wildlife monitoring, conservation teams review thousands of camera-trap videos to find the clips that contain red squirrels. This project helps reduce that manual workload by using computer vision to scan video folders, flag red squirrel sightings, and save useful detection outputs for review.

The pipeline can also be trained to recognise visually similar species, including grey squirrels, pine martens, and rats. This matters because animals such as rats can sometimes be mistaken for red squirrels, especially in low-light, blurry, or distant footage.

The command-line tools are used for dataset preparation, training, evaluation, and batch scanning. The Tkinter GUI is intended for easier day-to-day video scanning once a trained model is available.

---

## Contents

- [What this project does](#what-this-project-does)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Roboflow API key](#roboflow-api-key)
- [Quick start](#quick-start)
- [Colab notes](#colab-notes)
- [CLI workflows](#cli-workflows)
- [Training](#training)
- [Resume training](#resume-training)
- [Scanning videos](#scanning-videos)
- [Outputs](#outputs)
- [GUI scanning](#gui-scanning)
- [Build a Windows executable](#build-a-windows-executable)
- [Testing](#testing)
- [Data and security notes](#data-and-security-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## What this project does

Squirrel Scanner provides tools to:

- download public Roboflow object-detection datasets;
- merge different datasets into one consistent class list;
- train one or more Ultralytics detection models;
- continue interrupted training runs;
- scan wildlife video folders with a trained detector;
- save logs, metrics, weights, and scan outputs in run folders;
- provide a simple desktop interface for users who only need to scan videos.

The main classes currently used by the project are:

```text
red
grey
marten
rat
```

---

## Repository structure

```text
squirrel_scanner/
├── main.py                     # Command-line entry point
├── scanner_gui.py              # Desktop scanning interface
├── src/                        # Pipeline code
│   ├── acquisition.py          # Roboflow dataset download
│   ├── dataset_merge.py        # YOLO dataset merging and class remapping
│   ├── train_det.py            # Detector training flow
│   ├── train_det_ultra.py      # Ultralytics training backend
│   ├── scan_det_any.py         # Video scanning backend
│   ├── validate_dataset.py     # Dataset checks
│   ├── logger.py               # Run logging
│   └── experiment.py           # Run folder management
├── scripts/                    # Setup, checks, and release helpers
├── tests/                      # Lightweight tests
├── docs/                       # Supporting documentation
├── examples/                   # Example configs and manifests
├── requirements.txt            # Default requirements entry point
├── requirements-prod.txt       # Training and pipeline requirements
├── requirements-colab.txt      # Colab requirements
├── requirements-guiscan.txt    # GUI scanning requirements
├── .gitignore                  # Excludes data, weights, logs, and secrets
└── README.md
```

---

## Dataset

The datasets used by this project are hosted on Roboflow and downloaded when needed. Dataset files are not stored in this repository.

Current Roboflow dataset references:

```text
rollins/redz_greyz_martenz-95g9v/21
rollins/red-squirrels/2
```

A merged YOLO dataset is created locally and usually looks like this:

```text
merged_dataset/
├── data.yaml
├── merge_manifest.json
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Dataset licence

The datasets are publicly available on Roboflow. Public availability does not automatically define reuse rights, so check the licence shown on the Roboflow dataset page before redistribution or commercial reuse.

Dataset page: [Roboflow Universe](https://universe.roboflow.com/wildlife-yef4f/redz_greyz_martenz)

---

## Installation

### Local environment

```bash
git clone https://github.com/RollinsE/squirrel-scanner.git
cd squirrel-scanner
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-prod.txt
```

### Google Colab

Colab is useful for GPU training. The code does not depend on Colab, but the repository includes a Colab requirements file for convenience.

```bash
git clone https://github.com/RollinsE/squirrel-scanner.git
cd squirrel-scanner
pip install -r requirements-colab.txt
```

If you want trained weights and experiment outputs to survive after a Colab runtime disconnects, mount Google Drive and pass a Drive folder to `--run_base_dir`:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### GUI-only environment

```bash
pip install -r requirements-guiscan.txt
python scanner_gui.py
```

---

## Roboflow API key

Set your Roboflow API key as an environment variable. Do not commit API keys to GitHub.

Colab:

```python
import os
os.environ["ROBOFLOW_API_KEY"] = "<YOUR_ROBOFLOW_API_KEY>"
```

Linux/macOS terminal:

```bash
export ROBOFLOW_API_KEY="<YOUR_ROBOFLOW_API_KEY>"
```

Windows PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="<YOUR_ROBOFLOW_API_KEY>"
```

A safe template can be kept in `.env.example`, but real `.env` files are ignored by Git.

---

## Quick start

The commands below use local folders. These paths work on a normal machine and can be replaced with Colab paths when running in a notebook.

### 1. Acquire and merge datasets

```bash
python main.py \
  --mode acquire_multi \
  --roboflow_datasets \
    rollins/redz_greyz_martenz-95g9v/21 \
    rollins/red-squirrels/1 \
    rollins/rats-feqor-g6tbp/1 \
  --download_dir "data/datasets" \
  --merged_dataset_dir "data/merged_dataset" \
  --rf_format "yolov11" \
  --run_base_dir "runs"
```

`acquire_multi` downloads the datasets and creates the merged dataset. You do not need to run `merge_datasets` afterwards unless you want to rebuild the merged dataset with different inputs or class names.

### 2. Train two detector models

```bash
python main.py \
  --mode train_det \
  --raw_dataset "data/merged_dataset" \
  --models yolo11n,yolo11s \
  --epochs 60 \
  --imgsz 640 \
  --batch 16 \
  --optimizer auto \
  --lr 0.0001 \
  --weight_decay 0.0005 \
  --momentum 0.937 \
  --num_workers 2 \
  --run_base_dir "runs"
```

A good first pair is:

```text
yolo11n → faster and lighter
yolo11s → slightly stronger while still practical to train
```

### 3. Resume an interrupted run

```bash
python main.py \
  --mode train_det \
  --resume \
  --raw_dataset "data/merged_dataset" \
  --models yolo11n,yolo11s \
  --epochs 80 \
  --imgsz 640 \
  --run_base_dir "runs"
```

`--epochs` is the target total epoch count, not the number of extra epochs.

---

## Colab notes

When running in Colab, the usual pattern is:

```text
/content/datasets                                      # temporary dataset storage
/content/drive/MyDrive/squirrel_scanner/experiments    # persistent run outputs
```

Use `/content/datasets` for downloaded and merged datasets. Use Google Drive for logs, model weights, metrics, and scan outputs if you want to keep them after the runtime ends.

Example Colab acquire-and-merge command:

```bash
python main.py \
  --mode acquire_multi \
  --roboflow_datasets \
    rollins/redz_greyz_martenz-95g9v/21 \
    rollins/red-squirrels/1 \
    rollins/rats-feqor-g6tbp/1 \
  --download_dir "/content/datasets" \
  --merged_dataset_dir "/content/datasets/merged_dataset" \
  --rf_format "yolov11" \
  --run_base_dir "/content/drive/MyDrive/squirrel_scanner/experiments"
```

Example Colab training command:

```bash
python main.py \
  --mode train_det \
  --raw_dataset "/content/datasets/merged_dataset" \
  --models yolo11n,yolo11s \
  --epochs 60 \
  --imgsz 640 \
  --batch 16 \
  --run_base_dir "/content/drive/MyDrive/squirrel_scanner/experiments"
```

---

## CLI workflows

### Acquire one Roboflow dataset

Use this when training from a single Roboflow project.

```bash
python main.py \
  --mode acquire \
  --workspace "rollins" \
  --project "redz_greyz_martenz-95g9v" \
  --version 21 \
  --download_dir "data/datasets" \
  --rf_format "yolov11" \
  --run_base_dir "runs"
```

### Acquire multiple Roboflow datasets

Each dataset is passed as:

```text
workspace/project/version
```

Example:

```bash
python main.py \
  --mode acquire_multi \
  --roboflow_datasets \
    rollins/redz_greyz_martenz-95g9v/21 \
    rollins/red-squirrels/1 \
    rollins/rats-feqor-g6tbp/1 \
  --download_dir "data/datasets" \
  --merged_dataset_dir "data/merged_dataset" \
  --rf_format "yolov11" \
  --run_base_dir "runs"
```

### Merge already-downloaded YOLO datasets

Use this only when the datasets already exist locally and you want to combine them without downloading again.

```bash
python main.py \
  --mode merge_datasets \
  --raw_datasets \
    "data/datasets/DATASET_1_FOLDER" \
    "data/datasets/DATASET_2_FOLDER" \
    "data/datasets/DATASET_3_FOLDER" \
  --merged_dataset_dir "data/merged_dataset" \
  --target_classes red,grey,marten,rat \
  --class_map "red squirrels=red,grey squirrels=grey,martens=marten,rats=rat"
```

`--target_classes` defines the final class list:

```text
0 = red
1 = grey
2 = marten
3 = rat
```

`--class_map` renames class names from the source datasets into the final names.

Example:

```bash
--class_map "red squirrels=red,grey squirrels=grey,martens=marten,rats=rat"
```

Meaning:

```text
red squirrels  → red
grey squirrels → grey
martens        → marten
rats           → rat
```

The merge step is strict. If a source dataset contains an unknown class name that is not already in `--target_classes` and is not mapped through `--class_map`, the command stops with a clear error.

Inspect downloaded dataset class names with:

```bash
find data/datasets -maxdepth 3 -name data.yaml -print -exec cat {} \;
```

---

## Training

### Train one model

```bash
python main.py \
  --mode train_det \
  --raw_dataset "data/merged_dataset" \
  --models yolo11n \
  --epochs 60 \
  --imgsz 640 \
  --run_base_dir "runs"
```

### Train multiple models

```bash
python main.py \
  --mode train_det \
  --raw_dataset "data/merged_dataset" \
  --models yolo11n,yolo11s \
  --epochs 60 \
  --imgsz 640 \
  --batch 16 \
  --run_base_dir "runs"
```

### Train with model-specific batch sizes

```bash
python main.py \
  --mode train_det \
  --raw_dataset "data/merged_dataset" \
  --models yolo11n,yolo11s,rtdetr-l \
  --model_batches yolo11n=16,yolo11s=8,rtdetr-l=2 \
  --epochs 60 \
  --imgsz 640 \
  --run_base_dir "runs"
```

### Common training options

```text
--models          Comma-separated model list or model weight paths
--epochs          Target total epoch count
--imgsz           Training image size
--batch           Default batch size
--model_batches   Per-model batch overrides
--optimizer       Ultralytics optimizer; use auto by default
--lr              Initial learning rate passed to Ultralytics as lr0
--weight_decay    Weight decay
--momentum        Momentum
--num_workers     DataLoader workers
--device          auto, cpu, 0, 0,1, etc.
```

---

## Resume training

Use `--resume` to continue an interrupted run.

```bash
python main.py \
  --mode train_det \
  --resume \
  --raw_dataset "data/merged_dataset" \
  --models yolo11n,yolo11s \
  --epochs 80 \
  --imgsz 640 \
  --run_base_dir "runs"
```

Resume behaviour:

```text
If a model stopped before the target epoch count, training continues.
If a model already reached the target epoch count, it is skipped.
If the new target epoch count is higher, completed models are extended.
```

Example:

```text
Previous run:
yolo11n = 60 epochs completed
yolo11s = 5 epochs completed

Resume with --epochs 80:
yolo11n continues from 60 to 80
yolo11s continues from 5 to 80

Resume with --epochs 50:
yolo11n is skipped because it already exceeded 50
yolo11s continues or skips depending on its completed epoch count
```

Resume uses the latest run folder under `--run_base_dir` unless `--run_dir` is provided.

Resume a specific run:

```bash
python main.py \
  --mode train_det \
  --resume \
  --run_dir "runs/run_YYYYMMDD_HHMMSS" \
  --raw_dataset "data/merged_dataset" \
  --models yolo11n,yolo11s \
  --epochs 80 \
  --imgsz 640
```

---

## Scanning videos

`scan_det` scans a folder of videos using the champion detector recorded during a training run.

```bash
python main.py \
  --mode scan_det \
  --detector_run_dir "runs/run_YYYYMMDD_HHMMSS" \
  --video_dir "videos" \
  --target_class red \
  --conf 0.25 \
  --scan_imgsz 960 \
  --frame_skip 1 \
  --run_base_dir "runs"
```

`--detector_run_dir` should point to the training run folder containing:

```text
artifacts/champion_detector.json
```

The current scanner is video-focused. Supported video extensions include common formats such as `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, and `.wmv`.

### Suggested settings for small, distant, or blurry squirrels

```text
--conf 0.20 to 0.25
--scan_imgsz 960
--frame_skip 1
```

Lower confidence can catch more animals, but may also increase false positives. A larger image size can help with small or distant animals, but uses more compute.

### Quick image prediction with Ultralytics

For folders of still images, use Ultralytics prediction directly:

```bash
yolo predict \
  model="runs/run_YYYYMMDD_HHMMSS/artifacts/detectors/yolo11s/train/weights/best.pt" \
  source="data/merged_dataset/test/images" \
  imgsz=960 \
  conf=0.25 \
  save=True
```

---

## Outputs

Each run creates a timestamped folder:

```text
runs/
└── run_YYYYMMDD_HHMMSS/
    ├── artifacts/
    │   ├── champion_detector.json
    │   └── detectors/
    ├── logs/
    │   └── run.log
    ├── metrics/
    │   └── detectors_summary.json
    ├── plots/
    └── scan_outputs/
```

Typical trained weights are stored under:

```text
artifacts/detectors/<MODEL_TAG>/train/weights/best.pt
```

Useful training outputs may include:

```text
confusion_matrix.png
confusion_matrix_normalized.png
results.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
```

For this project, the most important validation checks are:

```text
rat → red false positives
red → rat confusion
small red squirrel misses
blurred or distant squirrel misses
```

---

## GUI scanning

Start the GUI with:

```bash
python scanner_gui.py
```

The GUI is intended for scanning videos with an existing trained detector. Typical user inputs include:

- model weights;
- input video folder;
- output folder;
- target class;
- confidence threshold;
- frame skip settings.

Additional GUI notes are available in:

```text
docs/GUI_RELEASE.md
```

---

## Build a Windows executable

The desktop GUI can be packaged as a Windows executable for users who do not need to run the training pipeline.

Build instructions are available in:

```text
docs/WINDOWS_BUILD.md
```

Build artifacts are not committed to this repository.

---

## Testing

Run the tests:

```bash
pytest
```

Run the repository check script:

```bash
python scripts/doctor.py
```

The tests are intentionally lightweight. They check repository structure, safety rules, training configuration parsing, and dataset merge behaviour. They do not train models or require private runtime files.

---

## Data and security notes

Do not commit:

```text
.env
ROBOFLOW_API_KEY
datasets/
data/
experiments/
runs/
videos/
*.pt
*.pth
*.onnx
*.engine
*.tflite
```

This repository should contain source code, tests, documentation, and lightweight examples only.

Recommended storage locations:

```text
Roboflow                 Dataset hosting
Local or cloud storage   Experiment outputs and model weights
GitHub Releases          Approved release artifacts
Hugging Face             Optional model hosting
GitHub repository        Source code and documentation
```

Before publishing additional files, confirm that your organisation approves the release of:

- source code;
- public dataset references;
- screenshots;
- example outputs;
- trained model weights, if any are published separately.

---

## Troubleshooting

### `invalid choice: 'infer'`

The scanning mode is:

```bash
--mode scan_det
```

There is no `--mode infer` command in this CLI.

### `--video_dir is not a directory`

Create the folder or point `--video_dir` to an existing video folder:

```bash
mkdir -p videos
```

Then place videos in that folder before running `scan_det`.

### `champion_detector.json not found`

Point `--detector_run_dir` to the full training run directory, not directly to the model weights.

Correct:

```text
runs/run_YYYYMMDD_HHMMSS
```

Incorrect:

```text
runs/run_YYYYMMDD_HHMMSS/artifacts/detectors/yolo11s/train/weights/best.pt
```

### Model misses small or blurry squirrels

Try:

```text
--conf 0.20
--scan_imgsz 960
--frame_skip 1
```

Also add the missed examples to the next Roboflow dataset version as hard examples.

### Rats are still detected as red squirrels

Add more rat examples, especially:

```text
small rats
blurred rats
rats in squirrel-like poses
rats in similar backgrounds
night or low-light rats
```

Then retrain and inspect the normalized confusion matrix.

---

## Roadmap

Possible next improvements:

- image-folder scanning through the main CLI;
- richer scan summary reports;
- confidence-threshold sweep utilities;
- easier export of missed detections as hard examples;
- optional export to ONNX;
- release workflow for GUI builds;
- deployment notes for field or edge devices.

---

## Contributing

Contributions should keep datasets, videos, model weights, API keys, and local run outputs out of Git. Please run `pytest` before submitting changes.

---

## License

`<LICENSE>`
