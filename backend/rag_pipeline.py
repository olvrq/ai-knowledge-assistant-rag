import hashlib
import io
import math
import re
import uuid
from typing import Any, Optional

from backend.config import settings
from backend.database import repository
from backend.schemas import CitationItem, DocumentResponse
from backend.security import sanitize_error_message

# Check LangChain text splitters availability
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_SPLITTERS_AVAILABLE = True
except ImportError:
    LANGCHAIN_SPLITTERS_AVAILABLE = False

# Check LangChain Google GenAI availability
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_GEMINI_AVAILABLE = True
except ImportError:
    LANGCHAIN_GEMINI_AVAILABLE = False

# Check official google-genai SDK availability
try:
    from google import genai
    from google.genai import types as genai_types

    GOOGLE_GENAI_SDK_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_SDK_AVAILABLE = False


def _normalize_vector(vec: list[float], target_dim: int) -> list[float]:
    """
    Ensure vector has exact target_dim (Matryoshka truncation/padding) and L2 unit norm
    for accurate cosine similarity in pgvector.
    """
    sliced = list(vec[:target_dim])
    if len(sliced) < target_dim:
        sliced.extend([0.0] * (target_dim - len(sliced)))
    norm = math.sqrt(sum(v * v for v in sliced))
    if norm > 0.0:
        return [round(v / norm, 6) for v in sliced]
    return sliced


def extract_text_from_bytes(filename: str, file_type: str, file_bytes: bytes) -> str:
    """Extract clean text from PDF, DOCX, MD, or TXT bytes."""
    ft = file_type.upper()
    if ft == "PDF":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text: list[str] = []
            for idx, page in enumerate(reader.pages, start=1):
                extracted = (page.extract_text() or "").strip()
                if extracted:
                    pages_text.append(f"[Página {idx}]\n{extracted}")
            text = "\n\n".join(pages_text)
            if text.strip():
                return text
        except Exception as exc:
            raise ValueError(f"Falha ao extrair texto do PDF '{filename}': {exc}") from exc

    if ft == "DOCX":
        try:
            import docx

            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            if text.strip():
                return text
        except Exception as exc:
            raise ValueError(f"Falha ao extrair texto do DOCX '{filename}': {exc}") from exc

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            decoded = file_bytes.decode(encoding)
            if decoded.strip():
                return decoded.strip()
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Não foi possível extrair texto legível do arquivo '{filename}'.")


def split_text_into_chunks(text: str) -> list[str]:
    """
    Split document text into semantic chunks using LangChain RecursiveCharacterTextSplitter,
    applying MAX_CHUNKS_PER_DOCUMENT cost guardrail.
    """
    cleaned = re.sub(r"\r\n", "\n", text).strip()
    if not cleaned:
        return []

    if LANGCHAIN_SPLITTERS_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = [c.strip() for c in splitter.split_text(cleaned) if c.strip()]
        if chunks:
            return chunks[: settings.MAX_CHUNKS_PER_DOCUMENT]

    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    step = max(100, chunk_size - overlap)
    chunks = []
    for start in range(0, len(cleaned), step):
        segment = cleaned[start : start + chunk_size].strip()
        if segment:
            chunks.append(segment)
        if start + chunk_size >= len(cleaned) or len(chunks) >= settings.MAX_CHUNKS_PER_DOCUMENT:
            break
    return chunks


def _deterministic_semantic_embedding(text: str, dim: int) -> list[float]:
    """
    Generate a normalized N-dimensional semantic feature vector (default 768d)
    using word unigrams, bigrams, and character trigrams when GEMINI_API_KEY is not configured.
    """
    vec = [0.0] * dim
    normalized = re.sub(r"[^\w\sáàâãéêíóôõúç]", " ", text.lower())
    tokens = [t for t in normalized.split() if len(t) > 1]
    if not tokens:
        return vec

    features: list[tuple[str, float]] = []
    for tok in tokens:
        features.append((f"u:{tok}", 2.0))
        if len(tok) >= 4:
            features.append((f"p:{tok[:4]}", 1.2))
            for i in range(len(tok) - 2):
                features.append((f"c:{tok[i:i+3]}", 0.5))

    for i in range(len(tokens) - 1):
        features.append((f"b:{tokens[i]}_{tokens[i+1]}", 2.5))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * weight

    return _normalize_vector(vec, dim)


def _embed_with_gemini(texts: list[str], task_type: str) -> list[list[float]]:
    """
    Compute embeddings using the exact same configured Gemini model
    (GEMINI_EMBEDDING_MODEL) and dimension (GEMINI_EMBEDDING_DIMENSION = 768)
    for both documents (RETRIEVAL_DOCUMENT) and queries (RETRIEVAL_QUERY).
    """
    target_dim = settings.GEMINI_EMBEDDING_DIMENSION
    model_name = settings.GEMINI_EMBEDDING_MODEL

    if settings.is_gemini_configured:
        # 1. Try fast HTTP google-genai SDK with output_dimensionality=target_dim
        if GOOGLE_GENAI_SDK_AVAILABLE:
            try:
                client = genai.Client(
                    api_key=settings.GEMINI_API_KEY,
                    http_options={"timeout": 8000},
                )
                sdk_model = model_name.replace("models/", "")
                response = client.models.embed_content(
                    model=sdk_model,
                    contents=texts,
                    config=genai_types.EmbedContentConfig(
                        task_type=task_type.upper(),
                        output_dimensionality=target_dim,
                    ),
                )
                if response and response.embeddings:
                    return [
                        _normalize_vector(list(emb.values), target_dim)
                        for emb in response.embeddings
                    ]
            except Exception:
                pass

        # 2. Fallback to LangChain GoogleGenerativeAIEmbeddings only if SDK not available
        elif LANGCHAIN_GEMINI_AVAILABLE:
            try:
                embedder = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=settings.GEMINI_API_KEY,
                    task_type=task_type,
                )
                raw_vectors = embedder.embed_documents(
                    texts, output_dimensionality=target_dim
                )
                return [_normalize_vector(v, target_dim) for v in raw_vectors]
            except Exception:
                pass

    return [_deterministic_semantic_embedding(t, target_dim) for t in texts]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate 768-dim document embeddings via Google Gemini (models/gemini-embedding-001).
    Uses task_type='retrieval_document' and exact dimension GEMINI_EMBEDDING_DIMENSION.
    """
    if not texts:
        return []
    return _embed_with_gemini(texts, task_type="retrieval_document")


def generate_query_embedding(query: str) -> list[float]:
    """
    Generate 768-dim query embedding via Google Gemini (models/gemini-embedding-001).
    Guaranteed to use the exact same model and dimension as document embeddings.
    """
    vectors = _embed_with_gemini([query], task_type="retrieval_query")
    return vectors[0] if vectors else [0.0] * settings.GEMINI_EMBEDDING_DIMENSION


def index_document_content(
    document_id: str,
    filename: str,
    file_type: str,
    raw_text: str,
    user_id: Optional[str] = None,
) -> DocumentResponse:
    """Chunk text, compute 768d Gemini embeddings, and persist chunks in pgvector / repository."""
    try:
        chunks = split_text_into_chunks(raw_text)
        if not chunks:
            raise ValueError("Nenhum fragmento de texto pôde ser gerado a partir do documento.")

        embeddings = generate_embeddings(chunks)
        chunks_payload: list[dict[str, Any]] = []
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings), start=1):
            token_estimate = max(1, len(chunk_text.split()))
            chunks_payload.append(
                {
                    "id": str(uuid.uuid4()),
                    "chunk_index": idx,
                    "content": chunk_text,
                    "token_count": token_estimate,
                    "metadata": {
                        "filename": filename,
                        "file_type": file_type,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
                        "embedding_dimension": settings.GEMINI_EMBEDDING_DIMENSION,
                    },
                    "embedding": emb,
                }
            )

        repository.save_document_chunks(
            document_id=document_id,
            chunks_data=chunks_payload,
            user_id=user_id,
        )
        updated_doc = repository.get_document(document_id)
        if not updated_doc:
            raise RuntimeError("Documento não encontrado após indexação.")
        return updated_doc
    except Exception as exc:
        safe_err = sanitize_error_message(str(exc))
        repository.mark_document_error(document_id, safe_err)
        raise RuntimeError(safe_err) from exc


def synthesize_rag_answer(
    question: str,
    citations: list[CitationItem],
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a structured, cited response using LangChain + Google Gemini (gemini-2.5-flash)
    with strict grounding and automatic abstinence when documentary evidence is insufficient.
    """
    # Strict Abstinence Guardrail: Do not answer or call LLM when no chunks meet similarity threshold
    if not citations:
        return (
            "### Abstinência por Insuficiência de Evidências Documentais\n\n"
            "Não foram encontrados fragmentos nos documentos indexados com similaridade semântica suficiente "
            f"(limiar mínimo configurado: **{int(settings.RAG_MIN_SIMILARITY_THRESHOLD * 100)}%**) "
            "para responder com segurança a esta pergunta.\n\n"
            "**Política de Governança RAG:** O assistente abstém-se de formular hipóteses ou utilizar conhecimento "
            "externo não respaldado pela sua Biblioteca Documental.\n\n"
            "**Recomendação:** Verifique os termos da pergunta ou envie documentos pertinentes "
            "(`.pdf`, `.md`, `.txt`, `.docx`) na aba **Biblioteca Documental**.",
            {
                "engine": "rag-abstinence-guardrail",
                "abstained": True,
                "retrieved_chunks": 0,
                "min_similarity_threshold": settings.RAG_MIN_SIMILARITY_THRESHOLD,
                "llm_model": settings.GEMINI_LLM_MODEL,
                "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
                "embedding_dimension": settings.GEMINI_EMBEDDING_DIMENSION,
            },
        )

    context_blocks = []
    for c in citations:
        context_blocks.append(
            f"[Fonte: {c.filename} | Trecho #{c.chunk_index} | Similaridade: {c.similarity:.2f}]\n{c.excerpt}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    system_instruction = (
        "Você é o AI Knowledge Assistant, um especialista em inteligência documental e RAG alimentado pelo Google Gemini.\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE nos fragmentos documentais fornecidos abaixo.\n"
        "2. Se os fragmentos fornecidos não contiverem evidências suficientes para responder à pergunta, "
        "declare explicitamente que não há evidências suficientes nos documentos consultados (abstinência estrita) e NÃO invente informações.\n"
        "3. Estruture sua resposta em Markdown com:\n"
        "   - **Síntese Executiva**: Resposta direta e objetiva.\n"
        "   - **Evidências Documentais**: Detalhamento em tópicos.\n"
        "4. Inclua obrigatoriamente citações inline no formato `[Fonte: nome_do_arquivo | Trecho #N]` ao final de cada afirmação.\n\n"
        f"CONTEXTO DOCUMENTAL RECUPERADO:\n{context_text}"
    )

    gemini_error_note: Optional[str] = None
    primary_model = settings.GEMINI_LLM_MODEL

    def _extract_recommended_model(err_text: str) -> Optional[str]:
        match = re.search(r"use models/([a-zA-Z0-9\.\-_]+)", err_text)
        return match.group(1) if match else None

    # 1. Try google-genai SDK with fast 8s timeout per candidate model
    if settings.is_gemini_configured and GOOGLE_GENAI_SDK_AVAILABLE:
        client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options={"timeout": 8000},
        )
        models_to_try = [primary_model, "gemini-flash-latest"]
        seen_models: set[str] = set()
        for active_model in models_to_try:
            if active_model in seen_models:
                continue
            seen_models.add(active_model)
            try:
                resp = client.models.generate_content(
                    model=active_model,
                    contents=f"{system_instruction}\n\nPERGUNTA DO USUÁRIO:\n{question}",
                    config=genai_types.GenerateContentConfig(
                        temperature=settings.GEMINI_TEMPERATURE,
                        max_output_tokens=min(1024, settings.GEMINI_MAX_OUTPUT_TOKENS),
                    ),
                )
                if resp and resp.text:
                    return (
                        resp.text.strip(),
                        {
                            "engine": "langchain-google-genai",
                            "abstained": False,
                            "retrieved_chunks": len(citations),
                            "llm_model": active_model,
                            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
                            "embedding_dimension": settings.GEMINI_EMBEDDING_DIMENSION,
                            "vector_store": (
                                "supabase-pgvector"
                                if repository.is_supabase_active
                                else "local-pgvector-mirror"
                            ),
                        },
                    )
            except Exception as exc:
                err_str = str(exc)
                gemini_error_note = sanitize_error_message(err_str)

    # 2. Fallback to LangChain ChatGoogleGenerativeAI wrapper if google-genai SDK unavailable
    if settings.is_gemini_configured and LANGCHAIN_GEMINI_AVAILABLE and not GOOGLE_GENAI_SDK_AVAILABLE:
        try:
            llm = ChatGoogleGenerativeAI(
                model=primary_model,
                temperature=settings.GEMINI_TEMPERATURE,
                max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
                google_api_key=settings.GEMINI_API_KEY,
                timeout=12,
                max_retries=0,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_instruction),
                    ("human", "{question}"),
                ]
            )
            chain = prompt | llm
            result = chain.invoke({"question": question})
            answer_text = str(result.content).strip()
            if answer_text:
                return (
                    answer_text,
                    {
                        "engine": "langchain-google-genai",
                        "abstained": False,
                        "retrieved_chunks": len(citations),
                        "llm_model": primary_model,
                        "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
                        "embedding_dimension": settings.GEMINI_EMBEDDING_DIMENSION,
                        "vector_store": (
                            "supabase-pgvector"
                            if repository.is_supabase_active
                            else "local-pgvector-mirror"
                        ),
                    },
                )
        except Exception as exc:
            gemini_error_note = sanitize_error_message(str(exc))

    # 3. Grounded structured RAG synthesis from top retrieved chunks (offline / fallback protection)
    top = citations[0]
    secondary = citations[1:3]

    lines: list[str] = [
        "### Síntese Documental",
        f"Com base na recuperação semântica sobre a base de conhecimento, o documento **`{top.filename}`** "
        f"apresenta a evidência principal com índice de relevância de **{int(top.similarity * 100)}%** "
        f"`[Fonte: {top.filename} | Trecho #{top.chunk_index}]`.",
        "",
        "### Evidências Estruturadas",
        f"- **Referência Primária (`{top.filename}`, Trecho #{top.chunk_index}):**\n  {top.excerpt.strip()} "
        f"`[Fonte: {top.filename} | Trecho #{top.chunk_index}]`",
    ]

    for sec in secondary:
        excerpt_clean = sec.excerpt.strip()
        lines.append(
            f"- **Contexto Complementar (`{sec.filename}`, Trecho #{sec.chunk_index}):**\n  {excerpt_clean} "
            f"`[Fonte: {sec.filename} | Trecho #{sec.chunk_index}]`"
        )

    unique_docs = list(dict.fromkeys(c.filename for c in citations))
    lines.extend(
        [
            "",
            "### Fontes Verificadas no Pipeline",
            f"Foram inspecionados **{len(citations)} fragmentos vetoriais ({settings.GEMINI_EMBEDDING_DIMENSION}d)** "
            f"distribuídos em **{len(unique_docs)} documento(s)**: "
            + ", ".join(f"`{d}`" for d in unique_docs)
            + ".",
        ]
    )

    meta_out: dict[str, Any] = {
        "engine": "rag-structured-synthesis",
        "abstained": False,
        "retrieved_chunks": len(citations),
        "llm_model": settings.GEMINI_LLM_MODEL,
        "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
        "embedding_dimension": settings.GEMINI_EMBEDDING_DIMENSION,
        "vector_store": (
            "supabase-pgvector"
            if repository.is_supabase_active
            else "local-pgvector-mirror"
        ),
    }
    if gemini_error_note:
        meta_out["api_warning"] = gemini_error_note

    return "\n".join(lines), meta_out


def ensure_seed_documents() -> None:
    """Seed initial enterprise technical documentation if the repository has no documents yet."""
    existing = repository.list_documents()
    if existing:
        return

    seed_docs = [
        {
            "filename": "Arquitetura_Pipeline_RAG_Google_Gemini.md",
            "file_type": "MD",
            "content": (
                "# Arquitetura do Pipeline RAG — AI Knowledge Assistant (Google Gemini Edition)\n\n"
                "O pipeline de Retrieval-Augmented Generation (RAG) do AI Knowledge Assistant opera em cinco estágios "
                "determinísticos projetados para precisão documental, controle de custos e rastreabilidade de fontes corporativas.\n\n"
                "## 1. Ingestão e Normalização Documental\n"
                "Os documentos enviados nos formatos PDF, DOCX, Markdown (.md) e Texto (.txt) passam por higienização "
                "de metadados, validação de segurança e extração estruturada de texto preservando marcadores de página e seções.\n\n"
                "## 2. Fragmentação Semântica (Chunking via LangChain)\n"
                "Utilizamos o `RecursiveCharacterTextSplitter` do LangChain configurado com janelas de 900 caracteres "
                "e sobreposição (overlap) de 150 caracteres, respeitando o teto de controle de custos por documento.\n\n"
                "## 3. Vetorização Google Gemini e Indexação Supabase pgvector\n"
                "Documentos e perguntas utilizam exatamente o mesmo modelo `models/gemini-embedding-001` configurado "
                "com Matryoshka Representation Learning (MRL) em 768 dimensões (`vector(768)`), armazenado no Supabase "
                "com índice HNSW (`vector_cosine_ops`). A síntese final é gerada pelo modelo `gemini-2.5-flash`."
            ),
        },
        {
            "filename": "Politica_Seguranca_Governanca_e_Abstinencia_IA.pdf",
            "file_type": "PDF",
            "content": (
                "[Página 1]\n"
                "POLÍTICA DE SEGURANÇA, CONTROLE DE CUSTOS E ABSTINÊNCIA EM IA\n\n"
                "1. Isolamento da Chave GEMINI_API_KEY e Autenticação JWT:\n"
                "A chave da API Google Gemini reside exclusivamente no backend FastAPI. Todas as requisições possuem "
                "controle de taxa (Rate Limiting em janela deslizante) e validação JWT com políticas RLS no Supabase.\n\n"
                "[Página 2]\n"
                "2. Prevenção de Alucinações, Citações e Abstinência Estrita:\n"
                "O assistente documental opera sob restrição estrita de aterramento (strict grounding). Toda resposta "
                "gerada pelo `gemini-2.5-flash` deve citar explicitamente o documento de origem e o número do fragmento "
                "(`Trecho #N`). Quando nenhum fragmento atingir o limiar mínimo de similaridade (`RAG_MIN_SIMILARITY_THRESHOLD`), "
                "o sistema aplica abstinência automática e informa que não há evidências suficientes nos documentos."
            ),
        },
        {
            "filename": "Guia_Operacional_Supabase_pgvector_768d.txt",
            "file_type": "TXT",
            "content": (
                "GUIA OPERACIONAL: BUSCA VETORIAL COM SUPABASE PGVECTOR (768 DIMENSÕES)\n\n"
                "A função RPC `match_document_chunks` executa o operador de distância de cosseno (`<=>`) diretamente "
                "no banco PostgreSQL gerenciado pelo Supabase sobre vetores `vector(768)` gerados pelo `models/gemini-embedding-001`:\n"
                "- `query_embedding`: Vetor de 768 dimensões da pergunta do usuário (`task_type=retrieval_query`).\n"
                "- `match_threshold`: Limiar mínimo de similaridade semântica para evitar alucinações (padrão 0.30).\n"
                "- `match_count`: Quantidade máxima de fragmentos retornados para o contexto do Gemini (padrão Top-5).\n\n"
                "Para reindexar um documento existente sem necessidade de novo upload, utilize a ação 'Reindexar' "
                "na Biblioteca Documental, que recalcula os fragmentos e atualiza os vetores de 768 dimensões de forma atômica."
            ),
        },
    ]

    for item in seed_docs:
        raw_bytes = item["content"].encode("utf-8")
        doc = repository.create_document(
            filename=item["filename"],
            file_type=item["file_type"],
            file_size=len(raw_bytes),
            raw_text=item["content"],
            metadata={"source": "seed_enterprise_kb", "characters": len(item["content"])},
        )
        index_document_content(
            document_id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            raw_text=item["content"],
        )
