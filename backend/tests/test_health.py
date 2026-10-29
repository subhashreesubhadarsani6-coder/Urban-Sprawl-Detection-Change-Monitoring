def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "timestamp" in data

def test_system_status(client):
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "engine_status" in data
    assert "active_gpu" in data
    assert "sentinel_api_connected" in data
