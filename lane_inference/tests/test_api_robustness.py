"""The API must not return a 500 for any input a user can supply."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lane_inference"))

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():
    from server.api import app
    return fastapi_testclient.TestClient(app)


def _jpg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


DEGENERATE = {
    "1x1": np.full((1, 1, 3), 120, np.uint8),
    "tiny": np.full((8, 8, 3), 120, np.uint8),
    "very_wide": np.full((40, 900, 3), 120, np.uint8),
    "very_tall": np.full((900, 40, 3), 120, np.uint8),
    "white": np.full((240, 320, 3), 255, np.uint8),
    "black": np.zeros((240, 320, 3), np.uint8),
}


@pytest.mark.parametrize("name", sorted(DEGENERATE))
def test_degenerate_image_is_not_a_server_error(client, name):
    r = client.post("/api/infer",
                    files={"file": (f"{name}.jpg", _jpg(DEGENERATE[name]), "image/jpeg")})
    assert r.status_code == 200, f"{name} returned {r.status_code}"
    body = r.json()
    assert "scene" in body

    if body.get("lane") is not None:
        assert body["scene"]["isRoad"] is True


@pytest.mark.parametrize("payload,expected", [
    (b"", 400),
    (b"this is not an image", 400),
    (bytes(range(256)) * 4, 400),
])
def test_undecodable_upload_is_a_client_error(client, payload, expected):
    r = client.post("/api/infer",
                    files={"file": ("x.jpg", payload, "image/jpeg")})
    assert r.status_code == expected, f"got {r.status_code}: {r.text[:120]}"


@pytest.mark.parametrize("value", ["0", "-5", "1e9", "nan", "not-a-number"])
def test_overlay_range_is_clamped(client, value):
    img = np.full((240, 320, 3), 120, np.uint8)
    r = client.post("/api/infer",
                    files={"file": ("f.jpg", _jpg(img), "image/jpeg")},
                    data={"maxRange": value})
    assert r.status_code in (200, 422), f"got {r.status_code}"


def test_clamp_range_rejects_non_finite():
    from server.api import clamp_range
    assert clamp_range(float("nan")) == 22.0
    assert clamp_range(float("inf")) == 22.0
    assert clamp_range(-5) == 5.0
    assert clamp_range(1e9) == 60.0
    assert clamp_range(20.0) == 20.0


def test_every_response_survives_a_nan():
    """The response class, not the individual payload, is what converts."""
    from server.api import SafeJSONResponse
    body = SafeJSONResponse({"a": float("nan"), "b": [1.0, float("inf")],
                             "c": {"d": float("-inf")}}).body
    assert b"NaN" not in body and b"Infinity" not in body
    assert b"null" in body
