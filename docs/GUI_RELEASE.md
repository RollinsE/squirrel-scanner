# GUI Release Guide

The recommended deployment method for non-technical users is a packaged Windows GUI attached to a GitHub Release.

## Release asset format

Use a ZIP file named like:

```text
SquirrelScanner-v1.0.0-Windows.zip
```

The ZIP should contain:

```text
SquirrelScanner.exe
README_RUN_APP.txt
```

## User instructions

The release ZIP should tell users to:

1. Download the latest Windows ZIP from the Releases page.
2. Unzip the folder.
3. Double-click `SquirrelScanner.exe`.
4. Select detector weights.
5. Select the input video folder.
6. Select an output folder.
7. Choose scan settings and start the scan.

## Build note

The executable should be built on Windows. PyInstaller is the usual packaging tool for this type of Tkinter application.

A typical build command is:

```bash
pyinstaller --onefile --windowed --name SquirrelScanner --icon squirrel.ico scanner_gui.py
```

For projects using PyTorch and Ultralytics, test the generated executable on a clean Windows machine before publishing.

## Package an already-built executable

After building `dist/SquirrelScanner.exe`, create a release ZIP:

```bash
python scripts/prepare_gui_release.py --exe dist/SquirrelScanner.exe --version v1.0.0
```

The generated ZIP in `release_builds/` can be uploaded to GitHub Releases.
