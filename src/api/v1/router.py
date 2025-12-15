"""V1 API router aggregation."""
from fastapi import APIRouter

from src.api.v1.endpoints import health, rag

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(rag.router, tags=["rag"])
