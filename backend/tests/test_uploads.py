import io

def test_valid_file_upload(client):
    file_content = b"SIMULATED_GEOTIFF_HEADER_CONTENT_BYTES"
    files = {"file": ("test_raster.tif", io.BytesIO(file_content), "image/tiff")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["extension"] == ".tif"
    assert "url" in data

def test_invalid_extension_rejected(client):
    file_content = b"MALICIOUS_SCRIPT_CONTENT"
    files = {"file": ("malicious.exe", io.BytesIO(file_content), "application/octet-stream")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_path_traversal_sanitization(client):
    file_content = b"TEST_TRAVERSAL"
    files = {"file": ("../../etc/passwd.tif", io.BytesIO(file_content), "image/tiff")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    # Ensure filename was sanitized and does not contain traversal dots
    assert ".." not in data["filename"]
    assert "passwd" in data["filename"] or "file_" in data["filename"]
