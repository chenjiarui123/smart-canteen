"""Load the separately versioned targeted strategy; preserve Trial 18 rollback."""
import hashlib
import builtins
import importlib.util
import json

from .model_loader import frozen_policy_copy, make_recommendations_with_history, model_status, model_import
from .paths import MODEL_ROOT


def active_config():
    path = MODEL_ROOT / "targeted_config.json"
    if not path.exists():
        return frozen_policy_copy()
    manifest = json.loads(
        (MODEL_ROOT / "targeted_manifest.json").read_text(encoding="utf-8-sig")
    )
    for name, expected in manifest["files"].items():
        if hashlib.sha256((MODEL_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"新策略完整性校验失败：{name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def generate(contexts, config, constraints, **kwargs):
    if not config.get("targeted_enabled"):
        return make_recommendations_with_history(contexts, config, constraints, **kwargs)
    spec = importlib.util.spec_from_file_location("smart_canteen_targeted_policy", MODEL_ROOT / "targeted_policy.py")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=model_import)
    spec.loader.exec_module(module)
    # Preserve existing runtime handling of future gaps. This is explicitly
    # separate from historical backtest candidate inactivity.
    return module.make_recommendations(contexts, dict(config, runtime_history_end_inactivity=True), constraints, **kwargs)


def active_status():
    status = model_status()
    if (MODEL_ROOT / "targeted_config.json").exists():
        active_config()
        manifest = json.loads(
            (MODEL_ROOT / "targeted_manifest.json").read_text(encoding="utf-8-sig")
        )
        status.update(model_version=manifest["model_version"], strategy_modified=True,
                      parameter_sha256=manifest["files"]["targeted_config.json"],
                      targeted_policy_sha256=manifest["files"]["targeted_policy.py"],
                      base_model_version=status["model_version"])
    return status
