import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from backend.config import settings
from backend.schemas import (
    CitationItem,
    ConversationResponse,
    DocumentChunkResponse,
    DocumentResponse,
    MessageResponse,
    UserInfo,
)
from backend.security import hash_password, verify_password


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class KnowledgeRepository:
    """
    Persistence and vector search layer for AI Knowledge Assistant (Google Gemini Edition).
    Reads and writes directly to Supabase PostgreSQL + pgvector (vector(768)) in production,
    with automatic local SQLite + 768-dim cosine vector store fallback for local development.
    Enforces multi-tenant user isolation (user_id filtering).
    """

    def __init__(self) -> None:
        self._supabase_client = None
        self._init_local_db()
        self._init_supabase()

    def _init_supabase(self) -> None:
        if not settings.is_supabase_configured:
            self._supabase_client = None
            return
        try:
            from supabase import create_client

            api_key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY
            self._supabase_client = create_client(settings.SUPABASE_URL, api_key)
        except Exception:
            self._supabase_client = None

    @property
    def is_supabase_active(self) -> bool:
        return self._supabase_client is not None

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(settings.SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_local_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'processing',
                    error_message TEXT NULL,
                    raw_text TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                """
            )
            sample_chunk = conn.execute(
                "SELECT embedding FROM document_chunks LIMIT 1"
            ).fetchone()
            if sample_chunk:
                try:
                    emb_vec = json.loads(sample_chunk["embedding"] or "[]")
                    if len(emb_vec) != settings.GEMINI_EMBEDDING_DIMENSION:
                        conn.execute("DELETE FROM document_chunks;")
                        conn.execute(
                            "DELETE FROM documents WHERE metadata LIKE '%seed_enterprise_kb%';"
                        )
                except Exception:
                    pass

            row = conn.execute(
                "SELECT id FROM users WHERE email = ?", ("admin@knowledge.ai",)
            ).fetchone()
            if not row:
                conn.execute(
                    """
                    INSERT INTO users (id, email, full_name, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        "admin@knowledge.ai",
                        "Administrador Knowledge AI",
                        hash_password("admin123"),
                        _utc_now_iso(),
                    ),
                )
            conn.commit()

    # -------------------------------------------------------------------------
    # AUTHENTICATION
    # -------------------------------------------------------------------------
    def register_user(self, full_name: str, email: str, password: str) -> UserInfo:
        email_norm = email.strip().lower()
        if self.is_supabase_active:
            try:
                res = self._supabase_client.auth.sign_up(
                    {
                        "email": email_norm,
                        "password": password,
                        "options": {"data": {"full_name": full_name.strip()}},
                    }
                )
                if res and res.user:
                    return UserInfo(
                        id=str(res.user.id),
                        email=email_norm,
                        full_name=full_name.strip(),
                        auth_provider="supabase",
                    )
            except Exception:
                pass

        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email_norm,)
            ).fetchone()
            if existing:
                raise ValueError("Já existe uma conta cadastrada com este e-mail.")
            user_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO users (id, email, full_name, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    email_norm,
                    full_name.strip(),
                    hash_password(password),
                    _utc_now_iso(),
                ),
            )
            conn.commit()
            return UserInfo(
                id=user_id,
                email=email_norm,
                full_name=full_name.strip(),
                auth_provider="local",
            )

    def authenticate_user(self, email: str, password: str) -> Optional[UserInfo]:
        email_norm = email.strip().lower()
        if self.is_supabase_active:
            try:
                res = self._supabase_client.auth.sign_in_with_password(
                    {"email": email_norm, "password": password}
                )
                if res and res.user:
                    meta = res.user.user_metadata or {}
                    return UserInfo(
                        id=str(res.user.id),
                        email=email_norm,
                        full_name=meta.get("full_name", email_norm),
                        auth_provider="supabase",
                    )
            except Exception:
                pass

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, email, full_name, password_hash FROM users WHERE email = ?",
                (email_norm,),
            ).fetchone()
            if not row:
                return None
            if not verify_password(password, row["password_hash"]):
                return None
            return UserInfo(
                id=row["id"],
                email=row["email"],
                full_name=row["full_name"],
                auth_provider="local",
            )

    # -------------------------------------------------------------------------
    # DOCUMENTS & CHUNKS
    # -------------------------------------------------------------------------
    def create_document(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        raw_text: str,
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> DocumentResponse:
        doc_id = str(uuid.uuid4())
        now = _utc_now_iso()
        meta_dict = metadata or {}

        if self.is_supabase_active:
            try:
                payload = {
                    "id": doc_id,
                    "user_id": user_id,
                    "filename": filename,
                    "file_type": file_type,
                    "file_size": file_size,
                    "chunk_count": 0,
                    "status": "processing",
                    "metadata": {**meta_dict, "raw_text": raw_text},
                    "created_at": now,
                    "updated_at": now,
                }
                self._supabase_client.table("documents").insert(payload).execute()
            except Exception:
                pass

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, user_id, filename, file_type, file_size,
                    chunk_count, status, error_message, raw_text,
                    metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 0, 'processing', NULL, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    user_id,
                    filename,
                    file_type,
                    file_size,
                    raw_text,
                    json.dumps(meta_dict, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.commit()

        return DocumentResponse(
            id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            chunk_count=0,
            status="processing",
            error_message=None,
            metadata=meta_dict,
            created_at=now,
            updated_at=now,
        )

    def save_document_chunks(
        self,
        document_id: str,
        chunks_data: list[dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> None:
        now = _utc_now_iso()

        expected_dim = settings.GEMINI_EMBEDDING_DIMENSION
        for item in chunks_data:
            emb = item.get("embedding", [])
            if len(emb) != expected_dim:
                raise ValueError(
                    f"Dimensão de embedding inválida ({len(emb)}). "
                    f"Esperado: {expected_dim} para o modelo {settings.GEMINI_EMBEDDING_MODEL}."
                )

        if self.is_supabase_active:
            try:
                self._supabase_client.table("document_chunks").delete().eq(
                    "document_id", document_id
                ).execute()
                supa_rows = []
                for item in chunks_data:
                    supa_rows.append(
                        {
                            "id": item["id"],
                            "document_id": document_id,
                            "user_id": user_id,
                            "chunk_index": item["chunk_index"],
                            "content": item["content"],
                            "token_count": item["token_count"],
                            "metadata": item["metadata"],
                            "embedding": item["embedding"],
                            "created_at": now,
                        }
                    )
                if supa_rows:
                    self._supabase_client.table("document_chunks").insert(supa_rows).execute()
                self._supabase_client.table("documents").update(
                    {
                        "chunk_count": len(chunks_data),
                        "status": "indexed",
                        "error_message": None,
                        "updated_at": now,
                    }
                ).eq("id", document_id).execute()
            except Exception:
                pass

        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
            )
            for item in chunks_data:
                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, user_id, chunk_index, content,
                        token_count, metadata, embedding, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["id"],
                        document_id,
                        user_id,
                        item["chunk_index"],
                        item["content"],
                        item["token_count"],
                        json.dumps(item["metadata"], ensure_ascii=False),
                        json.dumps(item["embedding"]),
                        now,
                    ),
                )
            conn.execute(
                """
                UPDATE documents
                SET chunk_count = ?, status = 'indexed', error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (len(chunks_data), now, document_id),
            )
            conn.commit()

    def mark_document_error(self, document_id: str, error_message: str) -> None:
        now = _utc_now_iso()
        if self.is_supabase_active:
            try:
                self._supabase_client.table("documents").update(
                    {"status": "error", "error_message": error_message, "updated_at": now}
                ).eq("id", document_id).execute()
            except Exception:
                pass

        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = 'error', error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (error_message, now, document_id),
            )
            conn.commit()

    def list_documents(self, user_id: Optional[str] = None) -> list[DocumentResponse]:
        if self.is_supabase_active:
            try:
                query = self._supabase_client.table("documents").select(
                    "id, user_id, filename, file_type, file_size, chunk_count, "
                    "status, error_message, metadata, created_at, updated_at"
                )
                if user_id:
                    query = query.or_(f"user_id.eq.{user_id},user_id.is.null")
                res = query.order("created_at", desc=True).execute()
                if res.data is not None:
                    docs_out: list[DocumentResponse] = []
                    for r in res.data:
                        meta = dict(r.get("metadata") or {})
                        meta.pop("raw_text", None)
                        docs_out.append(
                            DocumentResponse(
                                id=str(r["id"]),
                                filename=str(r["filename"]),
                                file_type=str(r["file_type"]),
                                file_size=int(r.get("file_size", 0)),
                                chunk_count=int(r.get("chunk_count", 0)),
                                status=str(r.get("status", "indexed")),
                                error_message=r.get("error_message"),
                                metadata=meta,
                                created_at=str(r["created_at"]),
                                updated_at=str(r["updated_at"]),
                            )
                        )
                    return docs_out
            except Exception:
                pass

        with self._get_conn() as conn:
            if user_id:
                rows = conn.execute(
                    """
                    SELECT id, filename, file_type, file_size, chunk_count,
                           status, error_message, metadata, created_at, updated_at
                    FROM documents
                    WHERE user_id = ? OR user_id IS NULL
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, filename, file_type, file_size, chunk_count,
                           status, error_message, metadata, created_at, updated_at
                    FROM documents
                    ORDER BY created_at DESC
                    """
                ).fetchall()

            return [
                DocumentResponse(
                    id=r["id"],
                    filename=r["filename"],
                    file_type=r["file_type"],
                    file_size=r["file_size"],
                    chunk_count=r["chunk_count"],
                    status=r["status"],
                    error_message=r["error_message"],
                    metadata=json.loads(r["metadata"] or "{}"),
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def get_document(
        self, document_id: str, user_id: Optional[str] = None
    ) -> Optional[DocumentResponse]:
        if self.is_supabase_active:
            try:
                res = (
                    self._supabase_client.table("documents")
                    .select(
                        "id, user_id, filename, file_type, file_size, chunk_count, "
                        "status, error_message, metadata, created_at, updated_at"
                    )
                    .eq("id", document_id)
                    .limit(1)
                    .execute()
                )
                if res.data:
                    r = res.data[0]
                    if user_id and r.get("user_id") not in (None, user_id):
                        return None
                    meta = dict(r.get("metadata") or {})
                    meta.pop("raw_text", None)
                    return DocumentResponse(
                        id=str(r["id"]),
                        filename=str(r["filename"]),
                        file_type=str(r["file_type"]),
                        file_size=int(r.get("file_size", 0)),
                        chunk_count=int(r.get("chunk_count", 0)),
                        status=str(r.get("status", "indexed")),
                        error_message=r.get("error_message"),
                        metadata=meta,
                        created_at=str(r["created_at"]),
                        updated_at=str(r["updated_at"]),
                    )
            except Exception:
                pass

        with self._get_conn() as conn:
            r = conn.execute(
                """
                SELECT id, user_id, filename, file_type, file_size, chunk_count,
                       status, error_message, metadata, created_at, updated_at
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
            if not r:
                return None
            if user_id and r["user_id"] not in (None, user_id):
                return None
            return DocumentResponse(
                id=r["id"],
                filename=r["filename"],
                file_type=r["file_type"],
                file_size=r["file_size"],
                chunk_count=r["chunk_count"],
                status=r["status"],
                error_message=r["error_message"],
                metadata=json.loads(r["metadata"] or "{}"),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )

    def get_document_raw_text(self, document_id: str) -> Optional[str]:
        if self.is_supabase_active:
            try:
                res = (
                    self._supabase_client.table("documents")
                    .select("metadata")
                    .eq("id", document_id)
                    .limit(1)
                    .execute()
                )
                if res.data:
                    meta = res.data[0].get("metadata") or {}
                    if meta.get("raw_text"):
                        return str(meta["raw_text"])
            except Exception:
                pass

        with self._get_conn() as conn:
            r = conn.execute(
                "SELECT raw_text FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            return r["raw_text"] if r else None

    def get_document_chunks(self, document_id: str) -> list[DocumentChunkResponse]:
        if self.is_supabase_active:
            try:
                res = (
                    self._supabase_client.table("document_chunks")
                    .select("id, document_id, chunk_index, content, token_count, metadata, created_at")
                    .eq("document_id", document_id)
                    .order("chunk_index", desc=False)
                    .execute()
                )
                if res.data is not None:
                    return [
                        DocumentChunkResponse(
                            id=str(r["id"]),
                            document_id=str(r["document_id"]),
                            chunk_index=int(r["chunk_index"]),
                            content=str(r["content"]),
                            token_count=int(r.get("token_count", 0)),
                            metadata=dict(r.get("metadata") or {}),
                            created_at=str(r["created_at"]),
                        )
                        for r in res.data
                    ]
            except Exception:
                pass

        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, chunk_index, content, token_count, metadata, created_at
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
            return [
                DocumentChunkResponse(
                    id=r["id"],
                    document_id=r["document_id"],
                    chunk_index=r["chunk_index"],
                    content=r["content"],
                    token_count=r["token_count"],
                    metadata=json.loads(r["metadata"] or "{}"),
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def delete_document(
        self, document_id: str, user_id: Optional[str] = None
    ) -> bool:
        if self.is_supabase_active:
            try:
                q = self._supabase_client.table("documents").delete().eq("id", document_id)
                if user_id:
                    q = q.or_(f"user_id.eq.{user_id},user_id.is.null")
                q.execute()
            except Exception:
                pass

        with self._get_conn() as conn:
            if user_id:
                cur = conn.execute(
                    "DELETE FROM documents WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
                    (document_id, user_id),
                )
            else:
                cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            return cur.rowcount > 0

    # -------------------------------------------------------------------------
    # VECTOR SIMILARITY SEARCH (PGVECTOR 768D + LOCAL COSINE ENGINE)
    # -------------------------------------------------------------------------
    def search_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
        min_similarity: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> list[CitationItem]:
        threshold = (
            min_similarity
            if min_similarity is not None
            else settings.RAG_MIN_SIMILARITY_THRESHOLD
        )

        if len(query_embedding) != settings.GEMINI_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Dimensão do vetor da pergunta ({len(query_embedding)}) difere da dimensão "
                f"configurada no pgvector ({settings.GEMINI_EMBEDDING_DIMENSION})."
            )

        # 1. Try Supabase pgvector RPC first when active
        if self.is_supabase_active:
            try:
                rpc_res = self._supabase_client.rpc(
                    "match_document_chunks",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": threshold,
                        "match_count": top_k * 2 if document_ids else top_k,
                        "filter_user_id": user_id,
                    },
                ).execute()
                if rpc_res.data is not None:
                    items: list[CitationItem] = []
                    for row in rpc_res.data:
                        doc_id_str = str(row["document_id"])
                        if document_ids and doc_id_str not in document_ids:
                            continue
                        sim_val = round(float(row["similarity"]), 4)
                        if sim_val < threshold:
                            continue
                        meta = row.get("metadata") or {}
                        items.append(
                            CitationItem(
                                chunk_id=str(row["id"]),
                                document_id=doc_id_str,
                                filename=str(meta.get("filename", "Documento")),
                                file_type=str(meta.get("file_type", "DOC")),
                                chunk_index=int(row["chunk_index"]),
                                similarity=sim_val,
                                excerpt=str(row["content"]),
                            )
                        )
                    return items[:top_k]
            except Exception:
                pass

        # 2. Local 768-dim vector cosine similarity search with strict threshold & user filtering
        with self._get_conn() as conn:
            sql = """
                SELECT
                    dc.id AS chunk_id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    dc.embedding,
                    d.filename,
                    d.file_type
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE d.status = 'indexed'
            """
            params: list[Any] = []
            if user_id:
                sql += " AND (d.user_id = ? OR d.user_id IS NULL)"
                params.append(user_id)
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                sql += f" AND dc.document_id IN ({placeholders})"
                params.extend(document_ids)

            rows = conn.execute(sql, params).fetchall()
            scored: list[CitationItem] = []
            for r in rows:
                emb = json.loads(r["embedding"] or "[]")
                sim = _cosine_similarity(query_embedding, emb)
                if sim >= threshold:
                    scored.append(
                        CitationItem(
                            chunk_id=r["chunk_id"],
                            document_id=r["document_id"],
                            filename=r["filename"],
                            file_type=r["file_type"],
                            chunk_index=r["chunk_index"],
                            similarity=round(max(0.0, sim), 4),
                            excerpt=r["content"],
                        )
                    )

            scored.sort(key=lambda x: x.similarity, reverse=True)
            return scored[:top_k]

    # -------------------------------------------------------------------------
    # CONVERSATIONS & MESSAGES
    # -------------------------------------------------------------------------
    def create_conversation(
        self, title: str, user_id: Optional[str] = None
    ) -> ConversationResponse:
        conv_id = str(uuid.uuid4())
        now = _utc_now_iso()

        if self.is_supabase_active:
            try:
                self._supabase_client.table("conversations").insert(
                    {
                        "id": conv_id,
                        "user_id": user_id,
                        "title": title,
                        "created_at": now,
                        "updated_at": now,
                    }
                ).execute()
            except Exception:
                pass

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, user_id, title, now, now),
            )
            conn.commit()

        return ConversationResponse(
            id=conv_id,
            title=title,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def list_conversations(self, user_id: Optional[str] = None) -> list[ConversationResponse]:
        if self.is_supabase_active:
            try:
                query = self._supabase_client.table("conversations").select(
                    "id, user_id, title, created_at, updated_at, messages(count)"
                )
                if user_id:
                    query = query.or_(f"user_id.eq.{user_id},user_id.is.null")
                res = query.order("updated_at", desc=True).execute()
                if res.data is not None:
                    out: list[ConversationResponse] = []
                    for r in res.data:
                        msg_meta = r.get("messages") or []
                        msg_count = (
                            int(msg_meta[0].get("count", 0))
                            if isinstance(msg_meta, list) and msg_meta
                            else 0
                        )
                        out.append(
                            ConversationResponse(
                                id=str(r["id"]),
                                title=str(r["title"]),
                                created_at=str(r["created_at"]),
                                updated_at=str(r["updated_at"]),
                                message_count=msg_count,
                            )
                        )
                    return out
            except Exception:
                pass

        with self._get_conn() as conn:
            if user_id:
                rows = conn.execute(
                    """
                    SELECT
                        c.id,
                        c.title,
                        c.created_at,
                        c.updated_at,
                        COUNT(m.id) AS message_count
                    FROM conversations c
                    LEFT JOIN messages m ON m.conversation_id = c.id
                    WHERE c.user_id = ? OR c.user_id IS NULL
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    """,
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        c.id,
                        c.title,
                        c.created_at,
                        c.updated_at,
                        COUNT(m.id) AS message_count
                    FROM conversations c
                    LEFT JOIN messages m ON m.conversation_id = c.id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    """
                ).fetchall()

            return [
                ConversationResponse(
                    id=r["id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]

    def get_conversation(
        self, conversation_id: str, user_id: Optional[str] = None
    ) -> Optional[ConversationResponse]:
        if self.is_supabase_active:
            try:
                res = (
                    self._supabase_client.table("conversations")
                    .select("id, user_id, title, created_at, updated_at")
                    .eq("id", conversation_id)
                    .limit(1)
                    .execute()
                )
                if res.data:
                    r = res.data[0]
                    if user_id and r.get("user_id") not in (None, user_id):
                        return None
                    return ConversationResponse(
                        id=str(r["id"]),
                        title=str(r["title"]),
                        created_at=str(r["created_at"]),
                        updated_at=str(r["updated_at"]),
                        message_count=0,
                    )
            except Exception:
                pass

        with self._get_conn() as conn:
            r = conn.execute(
                """
                SELECT
                    c.id,
                    c.user_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.id = ?
                GROUP BY c.id
                """,
                (conversation_id,),
            ).fetchone()
            if not r:
                return None
            if user_id and r["user_id"] not in (None, user_id):
                return None
            return ConversationResponse(
                id=r["id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["message_count"],
            )

    def delete_conversation(
        self, conversation_id: str, user_id: Optional[str] = None
    ) -> bool:
        if self.is_supabase_active:
            try:
                q = self._supabase_client.table("conversations").delete().eq("id", conversation_id)
                if user_id:
                    q = q.or_(f"user_id.eq.{user_id},user_id.is.null")
                q.execute()
            except Exception:
                pass

        with self._get_conn() as conn:
            if user_id:
                cur = conn.execute(
                    "DELETE FROM conversations WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
                    (conversation_id, user_id),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM conversations WHERE id = ?", (conversation_id,)
                )
            conn.commit()
            return cur.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[list[CitationItem]] = None,
    ) -> MessageResponse:
        msg_id = str(uuid.uuid4())
        now = _utc_now_iso()
        citation_list = citations or []
        citations_serialized = [c.model_dump() for c in citation_list]

        if self.is_supabase_active:
            try:
                self._supabase_client.table("messages").insert(
                    {
                        "id": msg_id,
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content,
                        "citations": citations_serialized,
                        "created_at": now,
                    }
                ).execute()
                self._supabase_client.table("conversations").update(
                    {"updated_at": now}
                ).eq("id", conversation_id).execute()
            except Exception:
                pass

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, citations, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    conversation_id,
                    role,
                    content,
                    json.dumps(citations_serialized, ensure_ascii=False),
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit()

        return MessageResponse(
            id=msg_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citation_list,
            created_at=now,
        )

    def get_messages(self, conversation_id: str) -> list[MessageResponse]:
        if self.is_supabase_active:
            try:
                res = (
                    self._supabase_client.table("messages")
                    .select("id, conversation_id, role, content, citations, created_at")
                    .eq("conversation_id", conversation_id)
                    .order("created_at", desc=False)
                    .execute()
                )
                if res.data is not None:
                    out: list[MessageResponse] = []
                    for r in res.data:
                        raw_cit = r.get("citations") or []
                        citations = [CitationItem(**c) for c in raw_cit]
                        out.append(
                            MessageResponse(
                                id=str(r["id"]),
                                conversation_id=str(r["conversation_id"]),
                                role=str(r["role"]),
                                content=str(r["content"]),
                                citations=citations,
                                created_at=str(r["created_at"]),
                            )
                        )
                    return out
            except Exception:
                pass

        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, citations, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
            messages: list[MessageResponse] = []
            for r in rows:
                raw_citations = json.loads(r["citations"] or "[]")
                citations = [CitationItem(**c) for c in raw_citations]
                messages.append(
                    MessageResponse(
                        id=r["id"],
                        conversation_id=r["conversation_id"],
                        role=r["role"],
                        content=r["content"],
                        citations=citations,
                        created_at=r["created_at"],
                    )
                )
            return messages


repository = KnowledgeRepository()
