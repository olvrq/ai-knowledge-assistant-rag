import base64
from collections import defaultdict, deque
import hashlib
import hmac
import json
import re
import secrets
import time
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError:
    def Depends(dep: Any = None) -> Any:  # type: ignore
        return dep

    class status:  # type: ignore
        HTTP_400_BAD_REQUEST = 400
        HTTP_401_UNAUTHORIZED = 401
        HTTP_404_NOT_FOUND = 404
        HTTP_413_REQUEST_ENTITY_TOO_LARGE = 413
        HTTP_422_UNPROCESSABLE_ENTITY = 422
        HTTP_429_TOO_MANY_REQUESTS = 429
        HTTP_500_INTERNAL_SERVER_ERROR = 500

    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code: int, detail: str, headers: Optional[dict[str, str]] = None) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
            self.headers = headers or {}

    class HTTPAuthorizationCredentials:  # type: ignore
        credentials: str = ""

    class HTTPBearer:  # type: ignore
        def __init__(self, auto_error: bool = False) -> None:
            self.auto_error = auto_error

from backend.config import settings
from backend.schemas import UserInfo

bearer_scheme = HTTPBearer(auto_error=False)

# Sliding-window rate limiter store: client_id -> deque of request timestamps
_rate_limit_buckets: dict[str, deque[float]] = defaultdict(deque)


def reset_rate_limiter() -> None:
    """Clear in-memory rate limit buckets (useful for automated tests)."""
    _rate_limit_buckets.clear()


def enforce_rate_limit(client_identifier: str = "global") -> None:
    """
    Enforce sliding-window rate limit (RATE_LIMIT_REQUESTS_PER_MINUTE) to control
    Google Gemini API usage and costs. Raises HTTP 429 when limit is exceeded.
    """
    max_requests = max(1, settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
    window_seconds = 60.0
    now = time.monotonic()

    bucket = _rate_limit_buckets[client_identifier]
    while bucket and (now - bucket[0]) > window_seconds:
        bucket.popleft()

    if len(bucket) >= max_requests:
        retry_after = max(1, int(window_seconds - (now - bucket[0])))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Limite de requisições excedido ({max_requests} req/min para controle de custos da API Gemini). "
                f"Aguarde {retry_after}s antes de tentar novamente."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)


def sanitize_error_message(raw_error: str) -> str:
    """Ensure API keys or sensitive tokens are never leaked in error messages."""
    sanitized = str(raw_error)
    if settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 6:
        sanitized = sanitized.replace(settings.GEMINI_API_KEY, "[REDACTED_GEMINI_KEY]")
    if settings.SUPABASE_SERVICE_ROLE_KEY and len(settings.SUPABASE_SERVICE_ROLE_KEY) > 6:
        sanitized = sanitized.replace(settings.SUPABASE_SERVICE_ROLE_KEY, "[REDACTED_SUPABASE_KEY]")
    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{25,}", "[REDACTED_GEMINI_KEY]", sanitized)
    return sanitized


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash password using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()
    return f"{salt}${derived}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against PBKDF2-HMAC-SHA256 hash."""
    if "$" not in stored_hash:
        return False
    salt, _ = stored_hash.split("$", 1)
    expected = hash_password(password, salt)
    return hmac.compare_digest(expected, stored_hash)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(user: UserInfo) -> str:
    """Create a signed HS256 JWT token."""
    header = {"alg": settings.JWT_ALGORITHM, "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "auth_provider": user.auth_provider,
        "iat": now,
        "exp": now + settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(
        settings.JWT_SECRET_KEY.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    sig_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Optional[UserInfo]:
    """Verify and decode a signed HS256 JWT token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            settings.JWT_SECRET_KEY.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None

        return UserInfo(
            id=str(payload["sub"]),
            email=str(payload["email"]),
            full_name=str(payload.get("full_name", payload["email"])),
            auth_provider=str(payload.get("auth_provider", "local")),
        )
    except Exception:
        return None


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[UserInfo]:
    """Return authenticated user if Bearer token is valid, otherwise None."""
    if not credentials or not credentials.credentials:
        return None
    return decode_access_token(credentials.credentials)


async def get_current_user(
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> UserInfo:
    """Enforce authentication on protected endpoints."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticação necessária. Faça login para acessar este recurso.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def sanitize_filename(filename: str) -> str:
    """Remove path traversal characters and sanitize uploaded filenames."""
    clean_name = Path(filename).name
    clean_name = re.sub(r"[^\w\s\-\.\(\)]", "_", clean_name)
    return clean_name.strip() or "documento.txt"


def validate_upload_file(filename: str, file_bytes: bytes) -> tuple[str, str]:
    """Validate extension and payload size for uploaded documents."""
    clean_name = sanitize_filename(filename)
    ext = Path(clean_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(settings.ALLOWED_EXTENSIONS)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato '{ext}' não suportado. Formatos permitidos: {allowed_str}.",
        )
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo excede o limite máximo de {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O arquivo enviado está vazio.",
        )
    file_type = ext.lstrip(".").upper()
    return clean_name, file_type


def validate_question_input(question: str) -> str:
    """Validate question length to control prompt token costs."""
    cleaned = question.strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A pergunta não pode estar vazia.",
        )
    if len(cleaned) > settings.RAG_MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"A pergunta excede o limite máximo de {settings.RAG_MAX_QUESTION_CHARS} caracteres "
                "definido para controle de custos de tokens."
            ),
        )
    return cleaned
