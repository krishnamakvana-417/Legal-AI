"""
Centralised application settings loaded from .env
"""
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Google Drive ─────────────────────────────────────────────────────────
    GOOGLE_DRIVE_FOLDER_ID: str = "12pQVjSxHpY7AS8Zeiu375nq9rYRvwpXN"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # ── Indian Kanoon ────────────────────────────────────────────────────────
    INDIAN_KANOON_API_TOKEN: str = ""
    INDIAN_KANOON_BASE_URL: str = "https://api.indiankanoon.org"

    # ── AnrakLegal ───────────────────────────────────────────────────────────
    ANRAK_API_KEY: str = ""
    ANRAK_BASE_URL: str = "https://api.anraklegal.com"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "legal_judgments"

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "nlpaueb/legal-bert-base-uncased"
    EMBEDDING_FALLBACK: str = "sentence-transformers/all-MiniLM-L6-v2"
    USE_FALLBACK_EMBEDDING: bool = False

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # ── Search ───────────────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 10

    # ── Paths ────────────────────────────────────────────────────────────────
    JUDGMENTS_DIR: str = "./data/judgments"
    PARSED_DIR: str = "./data/parsed"
    GRAPH_PERSIST_PATH: str = "./data/citation_graph.gpickle"

    # ── Backend Server ───────────────────────────────────────────────────────
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_ORIGIN: str = "http://localhost:3000"

    # ── Helpers ──────────────────────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        for d in [self.JUDGMENTS_DIR, self.PARSED_DIR, self.CHROMA_PERSIST_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def active_embedding_model(self) -> str:
        return self.EMBEDDING_FALLBACK if self.USE_FALLBACK_EMBEDDING else self.EMBEDDING_MODEL


settings = Settings()
