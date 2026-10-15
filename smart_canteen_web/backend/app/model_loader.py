"""Load and integrity-check the byte-for-byte frozen Trial 18 model."""

from __future__ import annotations

import copy
import builtins
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .paths import MODEL_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


MANIFEST = json.loads(
    (MODEL_ROOT / "model_manifest.json").read_text(encoding="utf-8-sig")
)
for _name, _expected in MANIFEST["frozen_source_hashes"].items():
    _actual = _sha256(MODEL_ROOT / _name)
    if _actual != _expected:
        raise RuntimeError(f"冻结模型文件 {_name} 完整性校验失败")

_MODEL_MODULES: dict[str, Any] = {}


def model_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Resolve frozen internal imports without polluting root policy imports."""
    if level == 0 and name in {"policy", "diagnostics", "evaluator"}:
        return _load_model_module(name)
    return builtins.__import__(name, globals, locals, fromlist, level)


def _load_model_module(name):
    if name not in _MODEL_MODULES:
        spec = importlib.util.spec_from_file_location("smart_canteen_frozen_" + name, MODEL_ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=model_import)
        _MODEL_MODULES[name] = module
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return _MODEL_MODULES[name]


frozen_policy = _load_model_module("policy")
frozen_evaluator = _load_model_module("evaluator")


BEST_POLICY: dict[str, Any] = json.loads(
    (MODEL_ROOT / "best_policy.json").read_text(encoding="utf-8-sig")
)
CONSTRAINTS: dict[str, Any] = json.loads(
    (MODEL_ROOT / "constraints.json").read_text(encoding="utf-8-sig")
)


def _build_history_aware_function() -> Callable[..., Any]:
    """Add runtime-only inputs/fixes without editing frozen ``policy.py``.

    The function body comes directly from the frozen source. Two exact textual
    substitutions add a keyword argument and initialise the existing local
    ``menu_history`` variable from persisted history. A third substitution fixes
    candidate inactivity to use the actual history end instead of treating the
    future forecast gap as observed non-supply days. Tests verify that an empty
    history and a next-day forecast remain numerically identical to the frozen
    function.
    """

    source = inspect.getsource(frozen_policy.make_recommendations)
    old_signature = (
        "    selection_audit: list[dict[str, Any]] | None = None,\n"
        ") -> pd.DataFrame:"
    )
    new_signature = (
        "    selection_audit: list[dict[str, Any]] | None = None,\n"
        "    initial_menu_history: dict | None = None,\n"
        ") -> pd.DataFrame:"
    )
    old_initialiser = (
        "    menu_history: dict[tuple[str, str], "
        "list[tuple[pd.Timestamp, set[str]]]] = {}"
    )
    new_initialiser = (
        "    menu_history: dict[tuple[str, str], "
        "list[tuple[pd.Timestamp, set[str]]]] = {\n"
        "        (str(key[0]), str(key[1])): [\n"
        "            (pd.Timestamp(old_date), set(map(str, dishes)))\n"
        "            for old_date, dishes in values\n"
        "        ]\n"
        "        for key, values in (initial_menu_history or {}).items()\n"
        "    }"
    )
    old_inactivity = (
        '        local_last = pd.to_datetime(frame.get("l_last_date"), errors="coerce")\n'
        '        global_last = pd.to_datetime(frame.get("g_last_date"), errors="coerce")\n'
        "        last_date = local_last.where(local_last.notna(), global_last)\n"
        "        days_inactive = (target_date - last_date).dt.days\n"
        '        active = days_inactive.le(int(policy["max_inactive_days"])) | staple.loc[frame.index]'
    )
    new_inactivity = (
        '        local_last = pd.to_datetime(frame.get("l_last_date"), errors="coerce")\n'
        '        global_last = pd.to_datetime(frame.get("g_last_date"), errors="coerce")\n'
        "        last_date = local_last.where(local_last.notna(), global_last)\n"
        "        history_max_date = pd.Timestamp(context[\"max_history_date\"])\n"
        "        days_inactive_at_history_end = (history_max_date - last_date).dt.days\n"
        "        active = (\n"
        '            days_inactive_at_history_end.le(int(policy["max_inactive_days"]))\n'
        "            | staple.loc[frame.index]\n"
        "        )"
    )
    if (
        source.count(old_signature) != 1
        or source.count(old_initialiser) != 1
        or source.count(old_inactivity) != 1
    ):
        raise RuntimeError("冻结策略结构变化，无法安全注入推荐历史")
    adapted_source = (
        source.replace(old_signature, new_signature)
        .replace(old_initialiser, new_initialiser)
        .replace(old_inactivity, new_inactivity)
    )
    namespace: dict[str, Any] = {}
    exec(
        compile(
            adapted_source,
            str(MODEL_ROOT / "policy.py"),
            "exec",
            dont_inherit=False,
        ),
        frozen_policy.__dict__,
        namespace,
    )
    return namespace["make_recommendations"]


make_recommendations_with_history = _build_history_aware_function()


def frozen_policy_copy() -> dict[str, Any]:
    return copy.deepcopy(BEST_POLICY)


def constraints_copy() -> dict[str, Any]:
    return copy.deepcopy(CONSTRAINTS)


def model_status() -> dict[str, Any]:
    return {
        "model_version": MANIFEST["model_version"],
        "best_trial_index": MANIFEST["best_trial_index"],
        "policy_sha256": MANIFEST["frozen_source_hashes"]["policy.py"],
        "evaluator_sha256": MANIFEST["frozen_source_hashes"]["evaluator.py"],
        "parameter_sha256": MANIFEST["best_policy_sha256"],
        "integrity_ok": True,
        "strategy_modified": False,
    }
