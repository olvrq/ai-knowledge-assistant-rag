import math
from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import _resolve_writable_data_dir, settings
from backend.database import repository
from backend.rag_pipeline import (
    ensure_seed_documents,
    generate_embeddings,
    generate_query_embedding,
    index_document_content,
    synthesize_rag_answer,
)
from backend.security import (
    HTTPException,
    enforce_rate_limit,
    reset_rate_limiter,
    sanitize_error_message,
)


class TestGeminiArchitectureAndDeployReadiness(unittest.TestCase):
    """
    Automated test suite verifying Google Gemini RAG architecture, security,
    and pre-publication readiness for GitHub, Vercel, and Streamlit Community Cloud.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_seed_documents()

    def setUp(self) -> None:
        reset_rate_limiter()

    def test_01_gemini_models_and_dimensions_configured(self) -> None:
        self.assertEqual(settings.GEMINI_LLM_MODEL, "gemini-2.5-flash")
        self.assertEqual(settings.GEMINI_EMBEDDING_MODEL, "models/gemini-embedding-001")
        self.assertEqual(settings.GEMINI_EMBEDDING_DIMENSION, 768)
        self.assertGreaterEqual(settings.RAG_MIN_SIMILARITY_THRESHOLD, 0.1)
        self.assertGreater(settings.RATE_LIMIT_REQUESTS_PER_MINUTE, 0)

    def test_02_pgvector_schema_matches_768_dimensions(self) -> None:
        schema_sql = (ROOT_DIR / "supabase" / "schema.sql").read_text(encoding="utf-8")
        self.assertIn("embedding vector(768)", schema_sql)
        self.assertIn("query_embedding vector(768)", schema_sql)
        self.assertNotIn("vector(1536)", schema_sql)

    def test_03_no_openai_dependencies_and_split_requirements_for_vercel(self) -> None:
        req_text = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("openai", req_text)
        self.assertIn("langchain-google-genai", req_text)
        self.assertIn("google-genai", req_text)
        # Root requirements.txt should not bundle streamlit into Vercel serverless function
        req_lines = [
            line.strip()
            for line in req_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertFalse(any(line.startswith("streamlit") for line in req_lines))

        # Frontend requirements.txt must include streamlit for Streamlit Community Cloud
        fe_req_text = (ROOT_DIR / "frontend" / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertIn("streamlit", fe_req_text)

        env_example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("OPENAI_", env_example)
        self.assertIn("GEMINI_API_KEY", env_example)
        self.assertIn("GEMINI_LLM_MODEL=gemini-2.5-flash", env_example)
        self.assertIn("GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001", env_example)
        self.assertIn("GEMINI_EMBEDDING_DIMENSION=768", env_example)

    def test_04_gemini_api_key_isolated_in_backend(self) -> None:
        frontend_app = (ROOT_DIR / "frontend" / "app.py").read_text(encoding="utf-8")
        frontend_client = (ROOT_DIR / "frontend" / "api_client.py").read_text(encoding="utf-8")
        self.assertNotIn("GEMINI_API_KEY", frontend_app)
        self.assertNotIn("GEMINI_API_KEY", frontend_client)
        self.assertNotIn("OPENAI", frontend_app.upper())

    def test_05_document_and_query_embeddings_share_exact_model_and_768_dimension(self) -> None:
        doc_vectors = generate_embeddings(
            ["O modelo gemini-embedding-001 gera vetores de 768 dimensões no pgvector."]
        )
        query_vector = generate_query_embedding(
            "Qual a dimensão dos vetores do gemini-embedding-001 no pgvector?"
        )
        self.assertEqual(len(doc_vectors), 1)
        self.assertEqual(len(doc_vectors[0]), 768)
        self.assertEqual(len(query_vector), 768)

        norm_doc = math.sqrt(sum(x * x for x in doc_vectors[0]))
        norm_query = math.sqrt(sum(x * x for x in query_vector))
        self.assertAlmostEqual(norm_doc, 1.0, places=4)
        self.assertAlmostEqual(norm_query, 1.0, places=4)

    def test_06_retrieval_citations_and_strict_abstinence(self) -> None:
        q_vec = generate_query_embedding(
            "Como funciona o pipeline RAG com gemini-embedding-001 em 768 dimensões no pgvector?"
        )
        citations = repository.search_similar_chunks(q_vec, top_k=3, min_similarity=0.15)
        self.assertGreater(len(citations), 0)
        answer, meta = synthesize_rag_answer(
            "Como funciona o pipeline RAG com gemini-embedding-001?", citations
        )
        self.assertFalse(meta.get("abstained", False))
        self.assertIn("[Fonte:", answer)
        self.assertEqual(meta.get("embedding_dimension"), 768)

        abstained_answer, abstained_meta = synthesize_rag_answer(
            "Qual a receita culinária de bolo de cenoura marciano?", []
        )
        self.assertTrue(abstained_meta.get("abstained"))
        self.assertEqual(abstained_meta.get("engine"), "rag-abstinence-guardrail")
        self.assertIn("Abstinência por Insuficiência de Evidências Documentais", abstained_answer)

    def test_07_rate_limiting_and_api_key_sanitization(self) -> None:
        fake_err = "Error calling Gemini with key AIzaSyB1234567890abcdefghijklmnopqrstuv"
        cleaned = sanitize_error_message(fake_err)
        self.assertNotIn("AIzaSyB1234567890", cleaned)
        self.assertIn("[REDACTED_GEMINI_KEY]", cleaned)

        limit = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        for _ in range(limit):
            enforce_rate_limit("test_client_bucket")
        with self.assertRaises(HTTPException) as ctx:
            enforce_rate_limit("test_client_bucket")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_08_github_gitignore_and_vercel_serverless_readiness(self) -> None:
        gitignore = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore)
        self.assertIn("data/", gitignore)
        self.assertIn(".streamlit/secrets.toml", gitignore)
        self.assertIn("*.db", gitignore)

        vercel_json = (ROOT_DIR / "vercel.json").read_text(encoding="utf-8")
        self.assertIn("api/index.py", vercel_json)
        self.assertTrue((ROOT_DIR / "api" / "index.py").exists())

        # Verify serverless writable directory resolves to temp dir without crashing
        serverless_dir = _resolve_writable_data_dir(ROOT_DIR, is_serverless=True)
        self.assertTrue(serverless_dir.exists())
        self.assertNotEqual(serverless_dir, ROOT_DIR / "data")

    def test_09_multi_user_isolation(self) -> None:
        doc_u1 = repository.create_document(
            filename="Doc_Privado_UserA.txt",
            file_type="TXT",
            file_size=120,
            raw_text="Documento confidencial exclusivo do Usuário A sobre projeto Alfa.",
            user_id="user-a-uuid",
        )
        index_document_content(
            document_id=doc_u1.id,
            filename=doc_u1.filename,
            file_type=doc_u1.file_type,
            raw_text="Documento confidencial exclusivo do Usuário A sobre projeto Alfa.",
            user_id="user-a-uuid",
        )
        try:
            docs_for_a = repository.list_documents(user_id="user-a-uuid")
            docs_for_b = repository.list_documents(user_id="user-b-uuid")
            ids_a = {d.id for d in docs_for_a}
            ids_b = {d.id for d in docs_for_b}
            self.assertIn(doc_u1.id, ids_a)
            self.assertNotIn(doc_u1.id, ids_b)
        finally:
            repository.delete_document(doc_u1.id, user_id="user-a-uuid")


if __name__ == "__main__":
    unittest.main()
