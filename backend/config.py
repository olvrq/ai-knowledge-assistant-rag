import os
from pathlib import Path
import tempfile

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _resolve_writable_data_dir(base_dir: Path, is_serverless: bool) -> Path:
    """
    Resolve a writable directory for temporary uploads and local SQLite fallback.
    In Vercel Serverless Functions (/var/task is read-only), uses /tmp automatically.
    """
    if is_serverless:
        target = Path(tempfile.gettempdir()) / "ai_knowledge_assistant"
        target.mkdir(parents=True, exist_ok=True)
        return target

    candidate = base_dir / "data"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ai_knowledge_assistant"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


class Settings:
    """
    Centralized configuration for FastAPI, Supabase, pgvector, LangChain and Google Gemini API.
    All model names, embedding dimensions, rate limits, CORS and serverless constraints
    are managed via environment variables.
    """

    APP_NAME: str = "AI Knowledge Assistant API"
    APP_VERSION: str = "2.0.0"

    # Environment & Serverless Detection
    IS_VERCEL: bool = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))
    APP_ENV: str = os.getenv(
        "APP_ENV", "production" if IS_VERCEL else "development"
    ).strip().lower()
    IS_PRODUCTION: bool = APP_ENV == "production" or IS_VERCEL

    # Server
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # CORS Origins (Restricts to official Streamlit Cloud domain and localhost)
    DEFAULT_CORS_ORIGINS: str = (
        "https://ai-knowledge-assistant-ragg.streamlit.app,"
        "http://localhost:8501,"
        "http://127.0.0.1:8501"
    )
    CORS_ALLOWED_ORIGINS_RAW: str = os.getenv(
        "CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS
    ).strip()

    # Supabase & pgvector
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "").strip()
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "").strip()
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    # Google Gemini API (Exclusively used in Backend)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash").strip()
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
    ).strip()
    GEMINI_EMBEDDING_DIMENSION: int = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

    # RAG Chunking, Abstinence & Cost Controls
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    RAG_MIN_SIMILARITY_THRESHOLD: float = float(
        os.getenv("RAG_MIN_SIMILARITY_THRESHOLD", "0.30")
    )
    RAG_MAX_QUESTION_CHARS: int = int(os.getenv("RAG_MAX_QUESTION_CHARS", "2000"))
    MAX_CHUNKS_PER_DOCUMENT: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "250"))
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(
        os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "30")
    )

    # Seed & Demo Controls (seeds initial enterprise docs only when repository is empty)
    ENABLE_SEED_DOCUMENTS: bool = (
        os.getenv("ENABLE_SEED_DOCUMENTS", "true").strip().lower()
        in ("1", "true", "yes")
    )
    DEMO_AUTO_LOGIN: bool = (
        os.getenv("DEMO_AUTO_LOGIN", "true").strip().lower()
        in ("1", "true", "yes")
    )

    # Security & Authentication
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY",
        "ai-knowledge-assistant-jwt-secret-key-change-in-production",
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

    # File Upload Constraints (Vercel Serverless limit is 4.5 MB per request payload)
    MAX_UPLOAD_SIZE_MB: int = int(
        os.getenv("MAX_UPLOAD_SIZE_MB", "4" if IS_VERCEL else "25")
    )
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md", ".docx")

    # Local storage directory (uses /tmp on Vercel Serverless)
    DATA_DIR: Path = _resolve_writable_data_dir(BASE_DIR, IS_VERCEL)
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    SQLITE_DB_PATH: Path = DATA_DIR / "knowledge_store.db"

    @property
    def cors_allowed_origins(self) -> list[str]:
        raw = self.CORS_ALLOWED_ORIGINS_RAW
        if not raw or (raw == "*" and self.IS_PRODUCTION):
            raw = self.DEFAULT_CORS_ORIGINS
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def is_supabase_configured(self) -> bool:
        return bool(
            self.SUPABASE_URL
            and (self.SUPABASE_SERVICE_ROLE_KEY or self.SUPABASE_KEY)
            and not self.SUPABASE_URL.startswith("https://your-project")
        )

    @property
    def is_gemini_configured(self) -> bool:
        return bool(
            self.GEMINI_API_KEY
            and not self.GEMINI_API_KEY.startswith("your-")
        )


settings = Settings()
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
