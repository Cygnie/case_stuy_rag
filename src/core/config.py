"""Core configuration using Pydantic BaseSettings."""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    llm_api_key: str
    embedding_api_key: str
    
    # Qdrant Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "ntt_hybrid_experiment"
    
    # Model Settings
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/embedding-001"
    llm_temperature: float = 0.7
    
    # RAG Settings
    rag_k: int = 5
    
    # System Settings
    log_level: str = "INFO"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    
settings = Settings()