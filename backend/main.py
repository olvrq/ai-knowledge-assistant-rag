from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import repository
from backend.rag_pipeline import (
    LANGCHAIN_SPLITTERS_AVAILABLE,
    ensure_seed_documents,
    extract_text_from_bytes,
    generate_query_embedding,
    index_document_content,
    synthesize_rag_answer,
)
from backend.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    ConversationCreateRequest,
    ConversationResponse,
    DashboardStatsResponse,
    DocumentChunkResponse,
    DocumentResponse,
    LoginRequest,
    MessageResponse,
    PipelineStatusResponse,
    RegisterRequest,
    TokenResponse,
    UserInfo,
)
from backend.security import (
    create_access_token,
    enforce_rate_limit,
    get_current_user,
    get_current_user_optional,
    sanitize_error_message,
    validate_question_input,
    validate_upload_file,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.ENABLE_SEED_DOCUMENTS:
        ensure_seed_documents()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Backend FastAPI para o AI Knowledge Assistant com Google Gemini API, "
        "Supabase, pgvector (768d) e LangChain."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=settings.cors_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_pipeline_status() -> PipelineStatusResponse:
    return PipelineStatusResponse(
        fastapi_status="online",
        supabase_connected=repository.is_supabase_active,
        pgvector_enabled=True,
        langchain_ready=LANGCHAIN_SPLITTERS_AVAILABLE,
        gemini_configured=settings.is_gemini_configured,
        storage_mode=(
            "Supabase + pgvector (768d)"
            if repository.is_supabase_active
            else "Persistência Vetorial Local (768d) + pgvector Ready"
        ),
        embedding_model=settings.GEMINI_EMBEDDING_MODEL,
        embedding_dimension=settings.GEMINI_EMBEDDING_DIMENSION,
        llm_model=settings.GEMINI_LLM_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        min_similarity_threshold=settings.RAG_MIN_SIMILARITY_THRESHOLD,
        rate_limit_rpm=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    )


def _resolve_client_key(request: Request, user: Optional[UserInfo]) -> str:
    if user and user.id:
        return f"user:{user.id}"
    client_host = request.client.host if request.client else "local"
    return f"ip:{client_host}"


# -----------------------------------------------------------------------------
# ROOT & HEALTH STATUS
# -----------------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root_status() -> dict[str, str]:
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }


@app.get("/api/health", response_model=PipelineStatusResponse, tags=["System"])
async def health_check() -> PipelineStatusResponse:
    return _build_pipeline_status()


# -----------------------------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------------------------
@app.post("/api/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(payload: RegisterRequest) -> TokenResponse:
    try:
        user = repository.register_user(
            full_name=payload.full_name,
            email=payload.email,
            password=payload.password,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    token = create_access_token(user)
    return TokenResponse(access_token=token, user=user)


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(payload: LoginRequest) -> TokenResponse:
    user = repository.authenticate_user(email=payload.email, password=payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas. Verifique seu e-mail e senha.",
        )
    token = create_access_token(user)
    return TokenResponse(access_token=token, user=user)


@app.get("/api/auth/me", response_model=UserInfo, tags=["Auth"])
async def get_me(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    return current_user


# -----------------------------------------------------------------------------
# DASHBOARD METRICS
# -----------------------------------------------------------------------------
@app.get("/api/dashboard/stats", response_model=DashboardStatsResponse, tags=["Dashboard"])
async def get_dashboard_stats(
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> DashboardStatsResponse:
    uid = user.id if user else None
    docs = repository.list_documents(user_id=uid)
    convs = repository.list_conversations(user_id=uid)

    total_docs = len(docs)
    indexed_docs = sum(1 for d in docs if d.status == "indexed")
    processing_docs = sum(1 for d in docs if d.status == "processing")
    error_docs = sum(1 for d in docs if d.status == "error")
    total_chunks = sum(d.chunk_count for d in docs)
    total_bytes = sum(d.file_size for d in docs)
    total_messages = sum(c.message_count for c in convs)

    format_breakdown: dict[str, int] = {}
    for d in docs:
        ft = d.file_type.upper()
        format_breakdown[ft] = format_breakdown.get(ft, 0) + 1

    return DashboardStatsResponse(
        total_documents=total_docs,
        indexed_documents=indexed_docs,
        processing_documents=processing_docs,
        error_documents=error_docs,
        total_chunks=total_chunks,
        total_conversations=len(convs),
        total_messages=total_messages,
        total_bytes=total_bytes,
        format_breakdown=format_breakdown,
        recent_documents=docs[:6],
        pipeline=_build_pipeline_status(),
    )


# -----------------------------------------------------------------------------
# DOCUMENT LIBRARY & INGESTION
# -----------------------------------------------------------------------------
@app.get("/api/documents", response_model=list[DocumentResponse], tags=["Documents"])
async def list_documents(
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> list[DocumentResponse]:
    return repository.list_documents(user_id=user.id if user else None)


@app.get("/api/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(
    document_id: str,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> DocumentResponse:
    doc = repository.get_document(document_id, user_id=user.id if user else None)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento não encontrado.",
        )
    return doc


@app.get(
    "/api/documents/{document_id}/chunks",
    response_model=list[DocumentChunkResponse],
    tags=["Documents"],
)
async def get_document_chunks(
    document_id: str,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> list[DocumentChunkResponse]:
    doc = repository.get_document(document_id, user_id=user.id if user else None)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento não encontrado.",
        )
    return repository.get_document_chunks(document_id)


@app.post("/api/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> DocumentResponse:
    enforce_rate_limit(_resolve_client_key(request, user))

    file_bytes = await file.read()
    clean_name, file_type = validate_upload_file(file.filename or "documento.txt", file_bytes)

    try:
        raw_text = extract_text_from_bytes(clean_name, file_type, file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=sanitize_error_message(str(exc)),
        ) from exc

    doc = repository.create_document(
        filename=clean_name,
        file_type=file_type,
        file_size=len(file_bytes),
        raw_text=raw_text,
        metadata={
            "characters": len(raw_text),
            "uploaded_by": user.email if user else "system",
            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
            "embedding_dimension": settings.GEMINI_EMBEDDING_DIMENSION,
        },
        user_id=user.id if user else None,
    )

    try:
        indexed_doc = index_document_content(
            document_id=doc.id,
            filename=clean_name,
            file_type=file_type,
            raw_text=raw_text,
            user_id=user.id if user else None,
        )
        return indexed_doc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro durante a indexação vetorial Gemini: {sanitize_error_message(str(exc))}",
        ) from exc


@app.post(
    "/api/documents/{document_id}/reindex",
    response_model=DocumentResponse,
    tags=["Documents"],
)
async def reindex_document(
    document_id: str,
    request: Request,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> DocumentResponse:
    enforce_rate_limit(_resolve_client_key(request, user))

    doc = repository.get_document(document_id, user_id=user.id if user else None)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento não encontrado.",
        )
    raw_text = repository.get_document_raw_text(document_id)
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conteúdo textual original não disponível para reindexação.",
        )

    return index_document_content(
        document_id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        raw_text=raw_text,
        user_id=user.id if user else None,
    )


@app.delete("/api/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> dict[str, str]:
    deleted = repository.delete_document(document_id, user_id=user.id if user else None)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento não encontrado.",
        )
    return {"status": "deleted", "document_id": document_id}


# -----------------------------------------------------------------------------
# CONVERSATIONS & RAG CHAT
# -----------------------------------------------------------------------------
@app.get("/api/conversations", response_model=list[ConversationResponse], tags=["Assistant"])
async def list_conversations(
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> list[ConversationResponse]:
    return repository.list_conversations(user_id=user.id if user else None)


@app.post("/api/conversations", response_model=ConversationResponse, tags=["Assistant"])
async def create_conversation(
    payload: ConversationCreateRequest,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> ConversationResponse:
    return repository.create_conversation(
        title=payload.title,
        user_id=user.id if user else None,
    )


@app.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
    tags=["Assistant"],
)
async def get_conversation_messages(
    conversation_id: str,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> list[MessageResponse]:
    conv = repository.get_conversation(conversation_id, user_id=user.id if user else None)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversa não encontrada.",
        )
    return repository.get_messages(conversation_id)


@app.delete("/api/conversations/{conversation_id}", tags=["Assistant"])
async def delete_conversation(
    conversation_id: str,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> dict[str, str]:
    deleted = repository.delete_conversation(
        conversation_id, user_id=user.id if user else None
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversa não encontrada.",
        )
    return {"status": "deleted", "conversation_id": conversation_id}


@app.post("/api/chat/query", response_model=ChatQueryResponse, tags=["Assistant"])
async def query_assistant(
    payload: ChatQueryRequest,
    request: Request,
    user: Optional[UserInfo] = Depends(get_current_user_optional),
) -> ChatQueryResponse:
    enforce_rate_limit(_resolve_client_key(request, user))
    question_clean = validate_question_input(payload.question)
    uid = user.id if user else None

    conv: Optional[ConversationResponse] = None
    if payload.conversation_id:
        conv = repository.get_conversation(payload.conversation_id, user_id=uid)

    if not conv:
        short_title = question_clean[:56] + ("..." if len(question_clean) > 56 else "")
        conv = repository.create_conversation(
            title=short_title,
            user_id=uid,
        )

    user_msg = repository.add_message(
        conversation_id=conv.id,
        role="user",
        content=question_clean,
        citations=[],
    )

    query_vec = generate_query_embedding(question_clean)
    citations = repository.search_similar_chunks(
        query_embedding=query_vec,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        min_similarity=settings.RAG_MIN_SIMILARITY_THRESHOLD,
        user_id=uid,
    )

    answer_text, pipeline_meta = synthesize_rag_answer(
        question=question_clean,
        citations=citations,
    )

    assistant_msg = repository.add_message(
        conversation_id=conv.id,
        role="assistant",
        content=answer_text,
        citations=citations,
    )

    return ChatQueryResponse(
        conversation_id=conv.id,
        conversation_title=conv.title,
        user_message=user_msg,
        assistant_message=assistant_msg,
        citations=citations,
        pipeline_metadata=pipeline_meta,
    )
