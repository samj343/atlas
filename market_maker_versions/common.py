import importlib.util
import pathlib
import sys

VERSIONS_DIR = pathlib.Path(__file__).parent

FILES = {
    "v1": VERSIONS_DIR / "mm_v1_baseline.py",
    "v2": VERSIONS_DIR / "mm_v2_model_pricer.py",
    "v3": VERSIONS_DIR / "mm_v3_risk_managed.py",
    "v4": VERSIONS_DIR / "mm_v4_adaptive.py",
    "v5": VERSIONS_DIR / "mm_v5_tuned.py",
    "v6": VERSIONS_DIR / "mm_v6_refit.py",
    "v7": VERSIONS_DIR / "mm_v7_v2refit.py",
}


def load_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_all():
    return {tag: load_module(f"mm_{tag}", path) for tag, path in FILES.items()}
