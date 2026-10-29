"""Unit and API integration tests for satellite imagery ingestion provider."""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.satellite_provider import (
    BoundingBox,
    MockSatelliteProvider,
    SentinelHubProvider,
    get_satellite_provider,
)

client = TestClient(app)


def test_bounding_box_validation():
    """Verify BoundingBox enforces valid WGS84 coordinates and ordering."""
    # Valid bounding box
    bbox = BoundingBox(min_lon=-115.35, min_lat=36.05, max_lon=-115.10, max_lat=36.25)
    assert bbox.to_list() == [-115.35, 36.05, -115.10, 36.25]

    # Out of latitude bounds
    with pytest.raises(ValueError, match="Latitude out of bounds"):
        BoundingBox(min_lon=0.0, min_lat=-95.0, max_lon=10.0, max_lat=20.0)

    # Inverted longitudes
    with pytest.raises(ValueError, match="min_lon .* must be less than max_lon"):
        BoundingBox(min_lon=10.0, min_lat=0.0, max_lon=5.0, max_lat=20.0)


def test_mock_provider_search_and_acquire(tmp_path):
    """Verify mock provider returns structured candidate scenes and stages pairs."""
    provider = MockSatelliteProvider()
    health = provider.get_health()
    assert health.connected is True
    assert "Mock" in health.provider_name

    bbox = BoundingBox(min_lon=-115.35, min_lat=36.05, max_lon=-115.10, max_lat=36.25)
    scenes = provider.search_scenes(bbox, date_from="2018-06-01", date_to="2024-06-01")
    assert len(scenes) == 2
    assert scenes[0].resolution_meters == 10.0
    assert "B02" in scenes[0].bands

    acquire_res = provider.acquire_scene_pair(
        t1_scene_id=scenes[0].scene_id,
        t2_scene_id=scenes[1].scene_id,
        bbox=bbox,
        output_dir=tmp_path
    )
    assert acquire_res["status"] == "success"
    assert acquire_res["is_live_download"] is False


def test_satellite_api_endpoints():
    """Verify HTTP API contracts for satellite provider health, search, and acquisition."""
    # 1. Health
    res_health = client.get("/api/v1/satellite/provider/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert "connected" in data_health
    assert "provider" in data_health

    # 2. Search with valid bounding box
    search_payload = {
        "min_lon": -115.35,
        "min_lat": 36.05,
        "max_lon": -115.10,
        "max_lat": 36.25,
        "date_from": "2018-06-01",
        "date_to": "2024-06-01",
        "max_cloud_cover": 10.0
    }
    res_search = client.post("/api/v1/satellite/search", json=search_payload)
    assert res_search.status_code == 200
    data_search = res_search.json()
    assert data_search["status"] == "success"
    assert data_search["count"] >= 1
    assert len(data_search["scenes"]) >= 1

    # 3. Search with invalid bounds returns HTTP 400
    invalid_payload = {
        "min_lon": 120.0,
        "min_lat": 36.05,
        "max_lon": 10.0,  # min_lon > max_lon
        "max_lat": 36.25,
        "date_from": "2018-06-01",
        "date_to": "2024-06-01"
    }
    res_invalid = client.post("/api/v1/satellite/search", json=invalid_payload)
    assert res_invalid.status_code == 400
