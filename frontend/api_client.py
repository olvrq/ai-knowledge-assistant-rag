import os
from typing import Any, Optional
import httpx


def _resolve_backend_url(explicit_url: Optional[str] = None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    env_url = os.getenv("BACKEND_API_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    try:
        import streamlit as st

        if "BACKEND_API_URL" in st.secrets:
            return str(st.secrets["BACKEND_API_URL"]).strip().rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:8000"


class KnowledgeAPIClient:
    """
    HTTP client connecting Streamlit UI to the FastAPI backend (local or Vercel).
    Reads BACKEND_API_URL from environment variables or Streamlit Community Cloud Secrets.
    When running locally without a separate Uvicorn process, transparently routes
    through FastAPI ASGI transport for zero-config local execution.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = _resolve_backend_url(base_url)
        self._seeded_local = False

    def _headers(self, token: Optional[str] = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        token: Optional[str] = None,
        json_data: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        headers = self._headers(token)
        try:
            with httpx.Client(base_url=self.base_url, timeout=45.0) as client:
                response = client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    json=json_data,
                    files=files,
                )
                return response
        except httpx.ConnectError:
            # Local development fallback: execute directly against FastAPI ASGI app
            from backend.config import settings
            from backend.main import app as fastapi_app
            from backend.rag_pipeline import ensure_seed_documents

            if not self._seeded_local and settings.ENABLE_SEED_DOCUMENTS:
                ensure_seed_documents()
                self._seeded_local = True

            transport = httpx.ASGITransport(app=fastapi_app)
            with httpx.Client(
                transport=transport, base_url="http://fastapi.internal", timeout=45.0
            ) as asgi_client:
                return asgi_client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    json=json_data,
                    files=files,
                )

    def login(self, email: str, password: str) -> dict[str, Any]:
        resp = self._request(
            "POST",
            "/api/auth/login",
            json_data={"email": email, "password": password},
        )
        resp.raise_for_status()
        return resp.json()

    def register(self, full_name: str, email: str, password: str) -> dict[str, Any]:
        resp = self._request(
            "POST",
            "/api/auth/register",
            json_data={"full_name": full_name, "email": email, "password": password},
        )
        resp.raise_for_status()
        return resp.json()

    def get_dashboard_stats(self, token: Optional[str] = None) -> dict[str, Any]:
        resp = self._request("GET", "/api/dashboard/stats", token=token)
        resp.raise_for_status()
        return resp.json()

    def list_documents(self, token: Optional[str] = None) -> list[dict[str, Any]]:
        resp = self._request("GET", "/api/documents", token=token)
        resp.raise_for_status()
        return resp.json()

    def get_document_chunks(
        self, document_id: str, token: Optional[str] = None
    ) -> list[dict[str, Any]]:
        resp = self._request(
            "GET", f"/api/documents/{document_id}/chunks", token=token
        )
        resp.raise_for_status()
        return resp.json()

    def upload_document(
        self,
        filename: str,
        file_bytes: bytes,
        content_type: str = "application/octet-stream",
        token: Optional[str] = None,
    ) -> dict[str, Any]:
        files = {"file": (filename, file_bytes, content_type)}
        resp = self._request(
            "POST", "/api/documents/upload", token=token, files=files
        )
        resp.raise_for_status()
        return resp.json()

    def reindex_document(
        self, document_id: str, token: Optional[str] = None
    ) -> dict[str, Any]:
        resp = self._request(
            "POST", f"/api/documents/{document_id}/reindex", token=token
        )
        resp.raise_for_status()
        return resp.json()

    def delete_document(
        self, document_id: str, token: Optional[str] = None
    ) -> dict[str, Any]:
        resp = self._request(
            "DELETE", f"/api/documents/{document_id}", token=token
        )
        resp.raise_for_status()
        return resp.json()

    def list_conversations(
        self, token: Optional[str] = None
    ) -> list[dict[str, Any]]:
        resp = self._request("GET", "/api/conversations", token=token)
        resp.raise_for_status()
        return resp.json()

    def create_conversation(
        self, title: str, token: Optional[str] = None
    ) -> dict[str, Any]:
        resp = self._request(
            "POST",
            "/api/conversations",
            token=token,
            json_data={"title": title},
        )
        resp.raise_for_status()
        return resp.json()

    def get_conversation_messages(
        self, conversation_id: str, token: Optional[str] = None
    ) -> list[dict[str, Any]]:
        resp = self._request(
            "GET", f"/api/conversations/{conversation_id}/messages", token=token
        )
        resp.raise_for_status()
        return resp.json()

    def delete_conversation(
        self, conversation_id: str, token: Optional[str] = None
    ) -> dict[str, Any]:
        resp = self._request(
            "DELETE", f"/api/conversations/{conversation_id}", token=token
        )
        resp.raise_for_status()
        return resp.json()

    def query_assistant(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
        token: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "conversation_id": conversation_id,
            "top_k": top_k,
            "document_ids": document_ids,
        }
        resp = self._request(
            "POST", "/api/chat/query", token=token, json_data=payload
        )
        resp.raise_for_status()
        return resp.json()


api_client = KnowledgeAPIClient()
