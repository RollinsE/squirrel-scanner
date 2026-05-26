import os
from datetime import datetime


def create_run_dir(base: str = "experiments"):
    """
    Creates a new run directory:
      <base>/run_YYYYMMDD_HHMMSS/
        logs/
        artifacts/
        metrics/
        plots/
    Returns: (run_id, run_dir)
    """
    os.makedirs(base, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{run_id}")

    for sub in ("logs", "artifacts", "metrics", "plots"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    return run_id, run_dir


def find_latest_run_dir(base: str = "experiments") -> str | None:
    """
    Returns the most recent run_<timestamp> folder under base, or None if none exist.
    """
    if not os.path.isdir(base):
        return None

    runs = []
    for name in os.listdir(base):
        if name.startswith("run_"):
            path = os.path.join(base, name)
            if os.path.isdir(path):
                runs.append(path)

    if not runs:
        return None

    runs.sort()
    return runs[-1]
