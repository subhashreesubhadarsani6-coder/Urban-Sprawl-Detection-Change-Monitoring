import time

def test_get_aois(client):
    response = client.get("/api/v1/aoi")
    assert response.status_code == 200
    aois = response.json()
    assert len(aois) >= 4
    assert any(a["id"] == "aoi-las-vegas" for a in aois)

def test_get_analysis_jobs(client):
    response = client.get("/api/v1/analysis")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) >= 2
    assert any(j["id"] == "job-vegas-2018-2024" for j in jobs)

def test_create_and_poll_analysis_job(client):
    payload = {
        "title": "Automated Test Sprawl Audit",
        "description": "Integration test analysis job",
        "aoiId": "aoi-las-vegas",
        "parameters": {
            "modelArch": "ChangeFormer (Transformer)",
            "confidenceThreshold": 0.85,
            "calculateNdvi": True,
            "calculateNdbi": True,
            "calculateMndwi": True,
            "cloudMaskTolerance": 10,
            "spatialFilteringRadius": 3
        }
    }
    create_res = client.post("/api/v1/analysis", json=payload)
    assert create_res.status_code == 202
    job_data = create_res.json()
    job_id = job_data["id"]
    assert job_id.startswith("job-")
    assert job_data["status"] == "queued"

    # Poll progress
    prog_res = client.get(f"/api/v1/analysis/{job_id}/progress")
    assert prog_res.status_code == 200
    prog_data = prog_res.json()
    assert "progressPercent" in prog_data
    assert "currentStep" in prog_data

def test_get_single_job_details(client):
    response = client.get("/api/v1/analysis/job-vegas-2018-2024")
    assert response.status_code == 200
    job = response.json()
    assert job["id"] == "job-vegas-2018-2024"
    assert job["status"] == "completed"
    assert "netExpansionKm2" in job["metrics"]
    assert job["metrics"]["netExpansionKm2"] > 0

def test_rethreshold_job_endpoint(client):
    response = client.post("/api/v1/analysis/job-vegas-2018-2024/rethreshold?threshold=0.7")
    assert response.status_code == 200
    updated = response.json()
    assert updated["parameters"]["confidenceThreshold"] == 0.7
    assert "changedPixels" in updated["metrics"]
    assert updated["metrics"]["changedPixels"] > 0

def test_export_csv_endpoint(client):
    response = client.get("/api/v1/analysis/job-vegas-2018-2024/export/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("Content-Type", "")
    assert "UrbanScope AI" in response.text
    assert "Net Expansion" in response.text

def test_get_nonexistent_job_returns_404(client):
    response = client.get("/api/v1/analysis/job-non-existent-999")
    assert response.status_code == 404

def test_get_datasets_catalog(client):
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    datasets = response.json()
    assert len(datasets) >= 4
    assert any("SpaceNet" in d["title"] or "Sentinel" in d["title"] for d in datasets)

def test_download_artifacts_and_path_traversal_protection(client):
    # Valid artifact download
    response = client.get("/api/v1/artifacts/job-vegas-2018-2024/audit_report.txt")
    assert response.status_code == 200
    assert "UrbanScope AI" in response.text

    # Path traversal protection
    bad_res = client.get("/api/v1/artifacts/job-vegas-2018-2024/..%2F..%2Fmain.py")
    assert bad_res.status_code in [400, 403, 404]

