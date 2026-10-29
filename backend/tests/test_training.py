def test_get_training_session(client):
    response = client.get("/api/v1/training/active")
    assert response.status_code == 200
    data = response.json()
    assert "train-run" in data["id"]
    assert data["currentEpoch"] >= 1
    assert data["totalEpochs"] >= 1
    assert len(data["history"]) >= 1

def test_toggle_pause_training(client):
    response = client.post("/api/v1/training/pause")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["paused", "running"]

def test_start_new_training(client):
    payload = {
        "modelArchitecture": "BIT-CD (Bitemporal Trans)",
        "backbone": "ResNet-50",
        "epochs": 40,
        "batchSize": 8,
        "learningRate": 0.0002,
        "optimizer": "AdamW",
        "lossFunction": "Combined (Focal + Dice)",
        "weightDecay": 0.01,
        "validationSplit": 0.2,
        "useDataAugmentation": True,
        "cropSize": 256
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["totalEpochs"] == 40
    assert data["config"]["backbone"] == "ResNet-50"

def test_get_benchmarks(client):
    response = client.get("/api/v1/evaluation/benchmarks")
    assert response.status_code == 200
    benchmarks = response.json()
    assert len(benchmarks) >= 4
    assert any(b["modelName"] == "ChangeFormer" for b in benchmarks)

def test_get_confusion_matrix(client):
    response = client.get("/api/v1/evaluation/confusion-matrix")
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 16
