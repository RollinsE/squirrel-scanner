from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_entrypoints_exist():
    assert (ROOT / "main.py").exists()
    assert (ROOT / "scanner_gui.py").exists()
    assert (ROOT / "src" / "scan_det_any.py").exists()


def test_documentation_files_exist():
    assert (ROOT / "README.md").exists()
    assert (ROOT / "docs" / "GUI_RELEASE.md").exists()
    assert (ROOT / "docs" / "MODEL_CARD.md").exists()
    assert (ROOT / "docs" / "DATA_CARD.md").exists()


def test_source_package_has_no_python_cache():
    assert not any((ROOT / "src").rglob("__pycache__"))



def test_current_training_module_exists():
    assert (ROOT / "src" / "train_det.py").exists()
