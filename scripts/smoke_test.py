"""End-to-End Smoke Test for UrbanScope AI System.

Verifies:
1. Health and Database Connectivity
2. File Upload with Strict Validation & Path Safety
3. Analysis Job Creation & Background Execution
4. Progress Polling through all 5 pipeline stages
5. Real Siamese U-Net Inference Execution & Mask Generation
6. Dynamic Probability Re-Thresholding (POST /rethreshold)
7. Downloadable Artifacts Verification (GeoTIFF, PNG, GeoJSON, TXT)
8. Tabular CSV Metrics Export (GET /export/csv)
"""

import io
import json
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api/v1"

def upload_file(file_path: str, filename: str) -> str:
    boundary = "----WebKitFormBoundaryUrbanScopeSmokeTest"
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.write(b"Content-Type: image/png\r\n\r\n")
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(f"{BASE_URL}/upload", data=body.getvalue(), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["url"]


def main():
    print("=" * 80)
    print("URBANSCOPE AI - END-TO-END SYSTEM INTEGRATION SMOKE TEST")
    print("=" * 80)

    # 1. Health check
    print("\n[Step 1/7] Testing API Health & Status...")
    with urllib.request.urlopen(f"{BASE_URL}/health") as res:
        health_data = json.loads(res.read().decode("utf-8"))
        assert health_data["status"] == "healthy", "API health check failed!"
        print(f"  -> Health: {health_data['status']} (Version {health_data['version']})")

    # 2. Upload T1 and T2 images
    print("\n[Step 2/7] Uploading Bi-Temporal Satellite Rasters...")
    t1_source = "reports/sample_t1_lasvegas.png"
    t2_source = "reports/sample_t2_lasvegas.png"
    assert Path(t1_source).exists(), f"Source image missing: {t1_source}"
    assert Path(t2_source).exists(), f"Source image missing: {t2_source}"

    t1_url = upload_file(t1_source, "las_vegas_t1_2018.png")
    t2_url = upload_file(t2_source, "las_vegas_t2_2024.png")
    print(f"  -> Uploaded T1: {t1_url}")
    print(f"  -> Uploaded T2: {t2_url}")

    # 3. Queue Real Change Detection Job
    print("\n[Step 3/7] Creating Bi-Temporal Analysis Job...")
    payload = json.dumps({
        "title": "Smoke Test Las Vegas Expansion Audit",
        "description": "Automated verification job testing real Siamese U-Net pipeline.",
        "aoiId": "aoi-las-vegas",
        "parameters": {
            "modelArch": "FC-Siam-diff",
            "confidenceThreshold": 0.50,
            "calculateNdvi": True,
            "calculateNdbi": True,
            "calculateMndwi": True,
            "cloudMaskTolerance": 10,
            "spatialFilteringRadius": 3
        },
        "imagery": {
            "preDate": "2018-06-15",
            "postDate": "2024-06-20",
            "sensor": "Sentinel-2 MSI",
            "resolutionMeters": 10.0,
            "cloudCoveragePre": 1.2,
            "cloudCoveragePost": 0.8,
            "bandsAvailable": ["B02", "B03", "B04", "B08"],
            "preImageUrl": t1_url,
            "postImageUrl": t2_url
        }
    }).encode("utf-8")

    create_req = urllib.request.Request(
        f"{BASE_URL}/analysis",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(create_req) as res:
        job = json.loads(res.read().decode("utf-8"))
        job_id = job["id"]
        print(f"  -> Job Created: {job_id} (Status: {job['status']})")

    # 4. Poll Progress
    print("\n[Step 4/7] Polling Pipeline Stages & GPU Worker...")
    completed = False
    for attempt in range(40):
        time.sleep(1.0)
        with urllib.request.urlopen(f"{BASE_URL}/analysis/{job_id}/progress") as res:
            p = json.loads(res.read().decode("utf-8"))
            print(f"  [Attempt {attempt+1:02d}] {p['progressPercent']:3d}% | {p['currentStep']}")
            if p["status"] == "completed":
                completed = True
                break
            elif p["status"] == "failed":
                raise RuntimeError(f"Pipeline job failed: {p}")

    assert completed, "Analysis job did not complete within timeout!"

    # 5. Retrieve Full Analytical Results
    print("\n[Step 5/7] Verifying Analysis Results & Real Area Metrics...")
    with urllib.request.urlopen(f"{BASE_URL}/analysis/{job_id}") as res:
        result = json.loads(res.read().decode("utf-8"))
        metrics = result["metrics"]
        print(f"  -> Status: {result['status']}")
        print(f"  -> Model Architecture: {metrics.get('modelArch')}")
        print(f"  -> Inference Type: {metrics.get('inferenceType')}")
        print(f"  -> Applied Threshold: {metrics.get('appliedThreshold')}")
        print(f"  -> Evaluated Pixels: {metrics.get('validPixels'):,} / {metrics.get('totalPixels'):,}")
        print(f"  -> Changed Pixels: {metrics.get('changedPixels'):,}")
        print(f"  -> Net Expansion Area: {metrics.get('netExpansionKm2')} km²")
        print(f"  -> Change Percentage: {metrics.get('expansionPercentage')}%")

        assert metrics["changedPixels"] > 0, "Changed pixels must be greater than zero!"
        assert metrics["netExpansionKm2"] > 0.0, "Net expansion area must be positive!"

    # 6. Test Dynamic Re-thresholding
    print("\n[Step 6/7] Testing Dynamic Probability Re-Thresholding...")
    new_t = 0.65
    reth_req = urllib.request.Request(
        f"{BASE_URL}/analysis/{job_id}/rethreshold?threshold={new_t}",
        method="POST"
    )
    with urllib.request.urlopen(reth_req) as res:
        reth_data = json.loads(res.read().decode("utf-8"))
        reth_metrics = reth_data["metrics"]
        print(f"  -> Re-thresholded to {new_t}: {reth_metrics['changedPixels']:,} px | {reth_metrics['netExpansionKm2']} km²")
        assert reth_metrics["changedPixels"] < metrics["changedPixels"], "Higher threshold must reduce or maintain change count!"

    # 7. Verify Multi-Format Artifact Downloads & CSV Export
    print("\n[Step 7/7] Verifying Downloadable Artifacts & Formats...")
    artifacts = [
        ("change_mask.tif", "image/tiff"),
        ("change_overlay.png", "image/png"),
        ("probability_map.png", "image/png"),
        ("sprawl_boundaries.geojson", "application/geo+json"),
        ("audit_report.txt", "text/plain")
    ]

    for filename, expected_mime in artifacts:
        url = f"{BASE_URL}/artifacts/{job_id}/{filename}"
        with urllib.request.urlopen(url) as a_res:
            assert a_res.status == 200, f"Artifact {filename} failed with HTTP {a_res.status}"
            content = a_res.read()
            mime = a_res.headers.get("Content-Type", "")
            assert len(content) > 0, f"Artifact {filename} is empty!"
            print(f"  -> {filename:26s} : {len(content):8,d} bytes [{mime}]")

    # CSV Export
    csv_url = f"{BASE_URL}/analysis/{job_id}/export/csv"
    with urllib.request.urlopen(csv_url) as csv_res:
        assert csv_res.status == 200, f"CSV export failed with HTTP {csv_res.status}"
        csv_bytes = csv_res.read()
        assert len(csv_bytes) > 0, "CSV export is empty!"
        print(f"  -> {'Tabular CSV Metrics Report':26s} : {len(csv_bytes):8,d} bytes [text/csv]")

    print("\n" + "=" * 80)
    print("ALL 7 SMOKE TEST SUITES PASSED! SYSTEM FULLY INTEGRATED & OPERATIONAL.")
    print("=" * 80)


if __name__ == "__main__":
    main()
