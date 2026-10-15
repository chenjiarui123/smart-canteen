import importlib.util
from pathlib import Path

import pytest

from smart_canteen_sdk import CanteenEngine


@pytest.fixture
def payload():
    path = Path(__file__).parents[1] / "examples" / "demo.py"
    spec = importlib.util.spec_from_file_location("sdk_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.demo_payload()


@pytest.fixture
def engine():
    return CanteenEngine()
