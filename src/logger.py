import logging
import os
import time
from contextlib import contextmanager


def setup_logger(run_id: str, run_dir: str, log_level: str = "INFO"):
    """
    Console + file logger with consistent formatting.
    """
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "run.log")

    logger = logging.getLogger("squirrel_scanner")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(run_id)s | squirrel_scanner | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RunIdFilter(logging.Filter):
        def filter(self, record):
            record.run_id = run_id
            return True

    logger.addFilter(RunIdFilter())

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info(f"Logger initialized. log_path={log_path}")
    logger.info("=" * 70)
    return logger


def log_stage(LOG, stage: str, component: str, **kwargs):
    """
    Standard structured log line.

    stage: START | INFO | WARN | DONE | FAIL | EPOCH | EVAL | CHECKPOINT | ARTIFACT | METRICS | RESULT
    component: Acquisition | Preprocess | YOLOv8n | FasterRCNN | RetinaNet | Scan | Pipeline
    """
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    msg = f"[{stage}] component={component}"
    if extras:
        msg += f" {extras}"
    LOG.info(msg)


@contextmanager
def component_stage(LOG, component: str, **kwargs):
    """
    Context manager that logs:
      [START] component=...
      [DONE]  component=... elapsed_sec=...
    and on exception:
      [FAIL]  component=... elapsed_sec=...
      plus stack trace.
    """
    log_stage(LOG, "START", component, **kwargs)
    t0 = time.time()
    try:
        yield
    except Exception:
        log_stage(LOG, "FAIL", component, elapsed_sec=f"{time.time() - t0:.1f}")
        LOG.exception(f"Exception in component={component}")
        raise
    else:
        log_stage(LOG, "DONE", component, elapsed_sec=f"{time.time() - t0:.1f}")
