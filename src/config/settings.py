"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration."""

    app_name: str = "Techno-AI"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    admin_api_key: str = Field(
        default="dev-key",
        alias="ADMIN_API_KEY",
    )
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="ALLOWED_ORIGINS",
    )
    # Active vector store: "pgvector" (default) or "chroma"
    vector_store_type: str = Field(default="pgvector", alias="VECTOR_STORE_TYPE")

    # Embedding output dimension — nomic-embed-text produces 768-dim vectors
    embedding_dimensions: int = Field(default=768, alias="EMBEDDING_DIMENSIONS")

    # ── Bot mode ─────────────────────────────────────────
    # Values: "wb" | "ah" | "technossus" (default)
    bot_mode: str = Field(default="technossus", alias="BOT_MODE")

    # ── Per-bot prefixed vars ─────────────────────────────
    WB_DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@db:5432/techno_ai",
        alias="WB_DATABASE_URL",
    )
    WB_WEBSITE_URL: str = Field(default="", alias="WB_WEBSITE_URL")
    WB_WEBSITE_URLS: str = Field(default="", alias="WB_WEBSITE_URLS")

    AH_DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@db:5432/techno_ai",
        alias="AH_DATABASE_URL",
    )
    AH_WEBSITE_URL: str = Field(default="", alias="AH_WEBSITE_URL")
    AH_WEBSITE_URLS: str = Field(default="", alias="AH_WEBSITE_URLS")

    # ── Canonical resolved vars ───────────────────────────
    # Default reads from unprefixed env vars (technossus case).
    # resolve_bot_mode() overrides these for "wb" and "ah".
    database_url: str = Field(
        default="postgresql://postgres:postgres@db:5432/techno_ai",
        alias="DATABASE_URL",
    )
    website_url: str = Field(default="", alias="WEBSITE_URL")
    website_urls: str = Field(default="", alias="WEBSITE_URLS")

    # ── Ingest ────────────────────────────────────────────
    enable_website_ingest: bool = Field(
        default=True,
        alias="ENABLE_WEBSITE_INGEST",
    )
    enable_file_ingest: bool = Field(
        default=False,
        alias="ENABLE_FILE_INGEST",
    )
    pdf_directory: str = Field(default="./data/pdfs", alias="PDF_DIRECTORY")
    chunk_size_words: int = Field(default=700, alias="CHUNK_SIZE_WORDS")
    chunk_overlap_words: int = Field(default=100, alias="CHUNK_OVERLAP_WORDS")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_min_score: float = Field(default=0.42, alias="RETRIEVAL_MIN_SCORE")
    request_timeout_seconds: int = 30
    user_agent: str = "Techno-AI/1.0"

    # ── Ollama ────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama3.2:8b", alias="OLLAMA_MODEL")
    ollama_rewrite_model: str = Field(
        default="gemma3:1b",
        alias="OLLAMA_REWRITE_MODEL",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        alias="OLLAMA_EMBEDDING_MODEL",
    )
    ollama_temperature: float = Field(default=0.2, alias="OLLAMA_TEMPERATURE")
    ollama_top_p: float = Field(default=0.8, alias="OLLAMA_TOP_P")
    ollama_num_predict: int = Field(default=180, alias="OLLAMA_NUM_PREDICT")

    # ── Groq ──────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_answer_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        alias="GROQ_ANSWER_MODEL",
    )
    groq_rewrite_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GROQ_REWRITE_MODEL",
    )

    # ── Nomic ─────────────────────────────────────────────
    nomic_api_key: str = Field(default="", alias="NOMIC_API_KEY")
    nomic_embedding_model: str = Field(
        default="nomic-embed-text-v1.5",
        alias="NOMIC_EMBEDDING_MODEL",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def resolve_bot_mode(self) -> "Settings":
        """Override canonical vars with the correct prefixed vars for BOT_MODE."""
        if self.bot_mode == "wb":
            self.database_url = self.WB_DATABASE_URL
            self.website_url = self.WB_WEBSITE_URL
            self.website_urls = self.WB_WEBSITE_URLS
        elif self.bot_mode == "ah":
            self.database_url = self.AH_DATABASE_URL
            self.website_url = self.AH_WEBSITE_URL
            self.website_urls = self.AH_WEBSITE_URLS
        # else: "technossus" or unrecognised — canonical fields already
        # populated from the unprefixed DATABASE_URL / WEBSITE_URL /
        # WEBSITE_URLS env vars via their Field aliases above.
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()  # type: ignore[call-arg]
