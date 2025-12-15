"""Integration tests for API endpoints."""
import pytest


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "RAG API"


def test_ask_endpoint_basic(client):
    """Test RAG ask endpoint with basic question."""
    response = client.post(
        "/api/v1/ask",
        json={"question": "What is NTT DATA?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "rewritten_question" in data
    assert len(data["answer"]) > 0


def test_ask_endpoint_with_year_extraction(client):
    """Test that years are automatically extracted."""
    response = client.post(
        "/api/v1/ask",
        json={"question": "2021-2023 carbon footprint data"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "years_extracted" in data


def test_ask_endpoint_validation(client):
    """Test input validation."""
    response = client.post(
        "/api/v1/ask",
        json={"question": ""}
    )
    assert response.status_code == 422
