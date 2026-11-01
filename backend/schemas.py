from typing import Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    def Field(default: Any = None, default_factory: Any = None, **_kwargs: Any) -> Any:  # type: ignore
        if default_factory is not None:
            return default_factory()
        return default

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "model_dump"):
                    out[k] = v.model_dump()
                elif isinstance(v, list):
                    out[k] = [x.model_dump() if hasattr(x, "model_dump") else x for x in v]
                else:
                    out[k] = v
            return out


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=4, max_length=128)


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class UserInfo(BaseModel):
    id: str
    email: str
    full_name: str
    auth_provider: str = "local"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class CitationItem(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    similarity: float
    excerpt: str


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="Nova Consulta Documental", min_length=1, max_length=200)


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    citations: list[CitationItem] = Field(default_factory=list)
    created_at: str


class ChatQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)
    conversation_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=15)
    document_ids: Optional[list[str]] = None


class ChatQueryResponse(BaseModel):
    conversation_id: str
    conversation_title: str
    user_message: MessageResponse
    assistant_message: MessageResponse
    citations: list[CitationItem]
    pipeline_metadata: dict[str, Any]


class PipelineStatusResponse(BaseModel):
    fastapi_status: str
    supabase_connected: bool
    pgvector_enabled: bool
    langchain_ready: bool
    gemini_configured: bool
    storage_mode: str
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    min_similarity_threshold: float
    rate_limit_rpm: int


class DashboardStatsResponse(BaseModel):
    total_documents: int
    indexed_documents: int
    processing_documents: int
    error_documents: int
    total_chunks: int
    total_conversations: int
    total_messages: int
    total_bytes: int
    format_breakdown: dict[str, int]
    recent_documents: list[DocumentResponse]
    pipeline: PipelineStatusResponse
