import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.seeding import seed_database

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    seed_database()

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
