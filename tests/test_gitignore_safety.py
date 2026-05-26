from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_blocks_large_runtime_files():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ["datasets/", "data/", "experiments/", "*.pt", "*.pth", "*.onnx", "*.engine", ".env"]:
        assert pattern in gitignore


def test_gui_numpy_requirement_preserved():
    req = (ROOT / "requirements-guiscan.txt").read_text(encoding="utf-8")
    assert "numpy>=2.3" in req
