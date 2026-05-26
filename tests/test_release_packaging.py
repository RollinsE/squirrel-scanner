from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_packaging_script_exists():
    script = ROOT / "scripts" / "prepare_gui_release.py"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "SquirrelScanner.exe" in text
    assert "GitHub Release" in text
