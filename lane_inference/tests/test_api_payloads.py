"""The results endpoint must always be JSON-encodable."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def api():
    return pytest.importorskip("server.api")


def test_finite_replaces_non_finite_floats(api):
    dirty = {"a": float("nan"), "b": [1.0, float("inf"), {"c": float("-inf")}], "d": 0.5}
    clean = api._finite(dirty)
    assert clean == {"a": None, "b": [1.0, None, {"c": None}], "d": 0.5}


def test_finite_leaves_ordinary_values_alone(api):
    payload = {"n": 1, "s": "text", "f": 0.25, "b": True, "z": None, "l": [1, 2]}
    assert api._finite(payload) == payload


def test_results_payload_is_encodable(api):
    """This is exactly what Starlette does when serialising the response."""
    payload = api.results()
    json.dumps(payload, allow_nan=False)


def test_results_carries_the_measured_tables(api):
    payload = api.results()
    assert payload["tables"]["evaluation_val"], "evaluation table is empty"
    assert payload["docs"], "no measurement records were loaded"


def test_source_results_files_still_contain_a_nan(api):
    """If this ever fails the guard above has gone untested, not that the bug is"""
    raw = (ROOT / "outputs" / "results" / "evaluation_val.json").read_text()
    values = json.loads(raw.replace("NaN", "null"))
    rows = values if isinstance(values, list) else values.get("evaluation_val", [])
    assert any(
        r.get("boundary_f1") is None or (
            isinstance(r.get("boundary_f1"), float) and math.isnan(r["boundary_f1"]))
        for r in rows
    ), "expected an undefined boundary metric among the trivial baselines"
