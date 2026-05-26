"""Package an already-built Squirrel Scanner GUI executable for GitHub Releases."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GitHub Release ZIP for the GUI executable.")
    parser.add_argument("--exe", required=True, help="Path to the built GUI executable, e.g. dist/SquirrelScanner.exe")
    parser.add_argument("--version", default="v1.0.0", help="Release version label, e.g. v1.0.0")
    parser.add_argument("--output-dir", default="release_builds", help="Folder where the ZIP will be written")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe_path = Path(args.exe).expanduser().resolve()
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    package_dir = output_dir / f"SquirrelScanner-{args.version}-Windows"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    shutil.copy2(exe_path, package_dir / "SquirrelScanner.exe")

    readme = package_dir / "README_RUN_APP.txt"
    readme.write_text(
        "Squirrel Scanner GUI\n"
        "=====================\n\n"
        "1. Unzip this folder.\n"
        "2. Double-click SquirrelScanner.exe.\n"
        "3. Select the trained detector weights file.\n"
        "4. Select the input video folder.\n"
        "5. Select an output folder.\n"
        "6. Choose scan settings.\n"
        "7. Click Start Scan.\n\n"
        "Model weights, videos, and outputs are not included in this package.\n",
        encoding="utf-8",
    )

    zip_path = output_dir / f"SquirrelScanner-{args.version}-Windows.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(package_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(package_dir.parent))

    print(f"Created release package: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
