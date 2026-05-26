# Windows Executable Build Guide

This guide is for maintainers who need to package the Squirrel Scanner desktop GUI as a Windows `.exe`.

The executable should be built on Windows because PyInstaller creates platform-specific binaries.

## Prerequisites

- Windows 10 or later
- Python 3.10 or later
- Git, if building from a cloned repository
- A tested detector weights file, if the release package should include one

## Install dependencies

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-guiscan.txt
pip install pyinstaller
```

## Build the executable

```bash
pyinstaller --onefile --windowed --name SquirrelScanner --icon squirrel.ico scanner_gui.py
```

The executable is created at:

```text
dist/SquirrelScanner.exe
```

## Test before release

Before publishing, test the executable on a clean Windows machine using representative model weights and sample videos. Confirm that the application can:

- open successfully;
- load detector weights;
- select an input video folder;
- create an output folder;
- complete a scan;
- save readable results.

## Prepare a release package

After the executable has been built and tested:

```bash
python scripts/prepare_gui_release.py --exe dist/SquirrelScanner.exe --version v1.0.0
```

The release package is written to:

```text
release_builds/
```

## Release guidance

Recommended approach:

- keep source code in the repository;
- attach approved Windows release ZIP files to GitHub Releases;
- do not commit `.exe`, `.pt`, `.onnx`, datasets, private videos, or experiment outputs directly to Git.
