import html
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.api_client import api_client

st.set_page_config(
    page_title="AI Knowledge Assistant — Inteligência Documental & RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_specify_css() -> None:
    css_path = Path(__file__).resolve().parent / "styles" / "specify_theme.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    if "active_view" not in st.session_state:
        st.session_state.active_view = "inicio"
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []
    if "last_pipeline_meta" not in st.session_state:
        st.session_state.last_pipeline_meta = {}
    if "inspected_doc_id" not in st.session_state:
        st.session_state.inspected_doc_id = None
    if "assistant_doc_filter" not in st.session_state:
        st.session_state.assistant_doc_filter = []
    if "show_auth_panel" not in st.session_state:
        st.session_state.show_auth_panel = False

    # Auto-authenticate default enterprise session in local/demo mode (controlled by DEMO_AUTO_LOGIN)
    demo_login_raw = os.getenv(
        "DEMO_AUTO_LOGIN",
        "false" if os.getenv("APP_ENV", "").lower() == "production" else "true",
    ).lower()
    demo_auto_login = demo_login_raw in ("1", "true", "yes")
    if (
        demo_auto_login
        and st.session_state.auth_token is None
        and not st.session_state.get("logged_out_explicitly")
    ):
        try:
            auth_data = api_client.login("admin@knowledge.ai", "admin123")
            st.session_state.auth_token = auth_data["access_token"]
            st.session_state.auth_user = auth_data["user"]
        except Exception:
            pass


def format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    kb = size_bytes / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024
    return f"{mb:.2f} MB"


def format_iso_date(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return iso_str[:16]


def status_badge_html(status_val: str) -> str:
    st_norm = (status_val or "").lower()
    if st_norm == "indexed":
        return (
            '<span class="ka-badge ka-badge-violet">'
            '<span class="ka-status-dot ka-status-dot-iris"></span>Indexado</span>'
        )
    if st_norm == "processing":
        return (
            '<span class="ka-badge ka-badge-neutral">'
            '<span class="ka-status-dot ka-status-dot-graphite"></span>Processando</span>'
        )
    return (
        '<span class="ka-badge ka-badge-dark">'
        '<span class="ka-status-dot ka-status-dot-graphite"></span>Erro</span>'
    )


def format_badge_html(file_type: str) -> str:
    ft = html.escape((file_type or "DOC").upper())
    return f'<span class="ka-badge ka-badge-neutral">{ft}</span>'


def navigate_to(view_name: str) -> None:
    st.session_state.active_view = view_name
    st.rerun()


# =============================================================================
# SIDEBAR — NAVEGAÇÃO LATERAL DISCRETA & AUTENTICAÇÃO
# =============================================================================
def render_sidebar(stats: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 8px 0 24px 0; border-bottom: 1px solid var(--color-mist); margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 28px; height: 28px; border-radius: 8px; background-color: var(--color-iris); display: flex; align-items: center; justify-content: center; color: #ffffff; font-weight: 700; font-size: 14px;">
                        ◈
                    </div>
                    <div>
                        <div style="font-size: 15px; font-weight: 700; color: var(--color-studio-slate); letter-spacing: -0.02em;">
                            Knowledge Assistant
                        </div>
                        <div style="font-size: 12px; color: var(--color-graphite); font-weight: 500;">
                            Enterprise RAG Platform
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--color-graphite); margin-bottom: 8px;">Navegação</div>',
            unsafe_allow_html=True,
        )

        nav_items = [
            ("inicio", "Página Inicial"),
            ("dashboard", "Dashboard"),
            ("biblioteca", "Biblioteca Documental"),
            ("assistente", "Assistente RAG"),
        ]

        for view_key, view_label in nav_items:
            is_active = st.session_state.active_view == view_key
            if st.button(
                view_label,
                key=f"nav_sidebar_{view_key}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                navigate_to(view_key)

        st.markdown(
            '<div style="margin-top: 24px; padding-top: 24px; border-top: 1px solid var(--color-mist);"></div>',
            unsafe_allow_html=True,
        )

        # Resumo rápido do Pipeline RAG na barra lateral
        pipeline = stats.get("pipeline", {})
        emb_dim = pipeline.get("embedding_dimension", 768)
        st.markdown(
            f"""
            <div class="ka-card" style="padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--color-iris); margin-bottom: 8px;">
                    Pipeline Ativo (Gemini)
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px;">
                    <span style="color: var(--color-iron);">Documentos</span>
                    <span style="font-weight: 600; color: var(--color-studio-slate);">{stats.get("indexed_documents", 0)} indexados</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px;">
                    <span style="color: var(--color-iron);">Fragmentos</span>
                    <span style="font-weight: 600; color: var(--color-studio-slate);">{stats.get("total_chunks", 0)} vetores</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span style="color: var(--color-iron);">Motor Vetorial</span>
                    <span style="font-weight: 600; color: var(--color-iris);">pgvector {emb_dim}d</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Painel de Autenticação e Segurança
        user = st.session_state.auth_user
        if user:
            st.markdown(
                f"""
                <div class="ka-card" style="padding: 16px;">
                    <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--color-graphite); margin-bottom: 6px;">
                        Sessão Autenticada (JWT)
                    </div>
                    <div style="font-size: 14px; font-weight: 600; color: var(--color-studio-slate);">
                        {html.escape(user.get("full_name", "Usuário"))}
                    </div>
                    <div style="font-size: 12px; color: var(--color-iron); margin-bottom: 8px;">
                        {html.escape(user.get("email", ""))}
                    </div>
                    <span class="ka-badge ka-badge-violet">Acesso Verificado</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Alternar Conta / Sair", key="btn_logout", type="secondary", use_container_width=True):
                st.session_state.auth_token = None
                st.session_state.auth_user = None
                st.session_state.logged_out_explicitly = True
                st.rerun()
        else:
            st.markdown(
                """
                <div class="ka-card" style="padding: 16px;">
                    <div style="font-size: 13px; font-weight: 600; color: var(--color-studio-slate); margin-bottom: 4px;">
                        Autenticação JWT / Supabase
                    </div>
                    <div style="font-size: 12px; color: var(--color-iron);">
                        Entre com suas credenciais corporativas para gerenciar documentos.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            auth_mode = st.radio(
                "Modo de Acesso",
                ["Entrar", "Criar Conta"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if auth_mode == "Entrar":
                with st.form("sidebar_login_form"):
                    email_in = st.text_input("E-mail", value="admin@knowledge.ai")
                    pass_in = st.text_input("Senha", type="password", value="admin123")
                    submit_login = st.form_submit_button("Autenticar", use_container_width=True)
                    if submit_login:
                        try:
                            res = api_client.login(email_in, pass_in)
                            st.session_state.auth_token = res["access_token"]
                            st.session_state.auth_user = res["user"]
                            st.session_state.logged_out_explicitly = False
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Falha na autenticação: {exc}")
            else:
                with st.form("sidebar_register_form"):
                    name_in = st.text_input("Nome completo")
                    email_in = st.text_input("E-mail corporativo")
                    pass_in = st.text_input("Senha (mín. 6 caracteres)", type="password")
                    submit_reg = st.form_submit_button("Registrar Conta", use_container_width=True)
                    if submit_reg:
                        try:
                            res = api_client.register(name_in, email_in, pass_in)
                            st.session_state.auth_token = res["access_token"]
                            st.session_state.auth_user = res["user"]
                            st.session_state.logged_out_explicitly = False
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Erro ao criar conta: {exc}")


# =============================================================================
# BARRA DE NAVEGAÇÃO SUPERIOR DISCRETA
# =============================================================================
def render_top_navigation() -> None:
    cols = st.columns([2.6, 1.1, 1.1, 1.4, 1.2])
    with cols[0]:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; padding-top: 4px;">
                <span class="ka-badge ka-badge-violet">
                    <span class="ka-status-dot ka-status-dot-iris"></span>RAG Engine v2.0
                </span>
                <span style="font-size: 13px; color: var(--color-graphite); font-weight: 500;">
                    FastAPI · Supabase · pgvector · LangChain · Google Gemini
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button(
            "Página Inicial",
            key="top_nav_inicio",
            type="primary" if st.session_state.active_view == "inicio" else "secondary",
            use_container_width=True,
        ):
            navigate_to("inicio")
    with cols[2]:
        if st.button(
            "Dashboard",
            key="top_nav_dashboard",
            type="primary" if st.session_state.active_view == "dashboard" else "secondary",
            use_container_width=True,
        ):
            navigate_to("dashboard")
    with cols[3]:
        if st.button(
            "Biblioteca Documental",
            key="top_nav_biblioteca",
            type="primary" if st.session_state.active_view == "biblioteca" else "secondary",
            use_container_width=True,
        ):
            navigate_to("biblioteca")
    with cols[4]:
        if st.button(
            "Assistente",
            key="top_nav_assistente",
            type="primary" if st.session_state.active_view == "assistente" else "secondary",
            use_container_width=True,
        ):
            navigate_to("assistente")

    st.markdown(
        '<div style="margin-top: 8px; margin-bottom: 24px; border-bottom: 1px solid var(--color-mist);"></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 1. PÁGINA INICIAL (HERO ESCURO + PIPELINE RAG + ACESSO AO ASSISTENTE)
# =============================================================================
def render_home_view(stats: dict[str, Any]) -> None:
    total_docs = stats.get("total_documents", 0)
    total_chunks = stats.get("total_chunks", 0)
    pipeline = stats.get("pipeline", {})

    # Hero escuro com identidade Knowledge Assistant, Headline em gradiente violeta,
    # descrição da solução e representação visual do pipeline RAG
    st.markdown(
        f"""
        <div class="ka-hero-section">
            <div class="ka-hero-badge">
                <span class="ka-hero-badge-dot"></span>
                KNOWLEDGE ASSISTANT · INTELIGÊNCIA DOCUMENTAL &amp; RAG
            </div>
            <h1 class="ka-hero-headline">
                Conhecimento documental transformado em
                <span class="ka-hero-headline-gradient">respostas precisas e auditáveis.</span>
            </h1>
            <p class="ka-hero-description">
                Arquitetura de Retrieval-Augmented Generation (RAG) construída com FastAPI, Supabase pgvector,
                LangChain e Google Gemini. Ingestão estruturada de documentos técnicos, indexação vetorial de alta
                densidade e síntese analítica com rastreabilidade exata de cada trecho citado.
            </p>
            <div class="ka-pipeline-wrapper">
                <div class="ka-pipeline-label">
                    Arquitetura Operacional do Pipeline RAG em Tempo Real
                </div>
                <div class="ka-pipeline-grid">
                    <div class="ka-pipeline-node ka-pipeline-node-active">
                        <div class="ka-pipeline-step">ETAPA 01</div>
                        <div class="ka-pipeline-title">Ingestão Documental</div>
                        <div class="ka-pipeline-meta">
                            Extração higienizada de PDF, DOCX, Markdown e TXT ({total_docs} documentos ativos).
                        </div>
                    </div>
                    <div class="ka-pipeline-node ka-pipeline-node-active">
                        <div class="ka-pipeline-step">ETAPA 02</div>
                        <div class="ka-pipeline-title">Chunking Semântico</div>
                        <div class="ka-pipeline-meta">
                            LangChain RecursiveSplitter ({pipeline.get("chunk_size", 900)} chars / overlap {pipeline.get("chunk_overlap", 150)}).
                        </div>
                    </div>
                    <div class="ka-pipeline-node ka-pipeline-node-active">
                        <div class="ka-pipeline-step">ETAPA 03</div>
                        <div class="ka-pipeline-title">Vetorização Gemini</div>
                        <div class="ka-pipeline-meta">
                            Embeddings MRL em {pipeline.get("embedding_dimension", 768)} dimensões via {html.escape(pipeline.get("embedding_model", "models/gemini-embedding-001"))}.
                        </div>
                    </div>
                    <div class="ka-pipeline-node ka-pipeline-node-active">
                        <div class="ka-pipeline-step">ETAPA 04</div>
                        <div class="ka-pipeline-title">Busca Supabase pgvector</div>
                        <div class="ka-pipeline-meta">
                            Similaridade de cosseno HNSW sobre {total_chunks} fragmentos indexados ({pipeline.get("embedding_dimension", 768)}d).
                        </div>
                    </div>
                    <div class="ka-pipeline-node ka-pipeline-node-active">
                        <div class="ka-pipeline-step">ETAPA 05</div>
                        <div class="ka-pipeline-title">Síntese com Citações</div>
                        <div class="ka-pipeline-meta">
                            Respostas estruturadas via {html.escape(pipeline.get("llm_model", "gemini-2.5-flash"))} com abstinência e citações.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Botões de ação diretos conectados às views funcionais
    cta_cols = st.columns([1.4, 1.4, 1.4, 2.8])
    with cta_cols[0]:
        if st.button("Acessar o Assistente →", key="hero_cta_assistant", type="primary", use_container_width=True):
            navigate_to("assistente")
    with cta_cols[1]:
        if st.button("Biblioteca Documental", key="hero_cta_library", type="secondary", use_container_width=True):
            navigate_to("biblioteca")
    with cta_cols[2]:
        if st.button("Explorar Dashboard", key="hero_cta_dashboard", type="secondary", use_container_width=True):
            navigate_to("dashboard")

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # Cards de Capacidades Arquiteturais na área branca interna
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="ka-card" style="height: 100%;">
                <div style="margin-bottom: 12px;">
                    <span class="ka-badge ka-badge-violet">Indexação Vetorial</span>
                </div>
                <h3 style="font-size: 18px; margin: 0 0 8px 0;">Supabase &amp; pgvector</h3>
                <p style="font-size: 14px; margin: 0;">
                    Armazenamento relacional e vetorial unificado com busca semântica por similaridade de cosseno
                    e políticas de Row Level Security (RLS).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="ka-card" style="height: 100%;">
                <div style="margin-bottom: 12px;">
                    <span class="ka-badge ka-badge-violet">Orquestração Cognitiva</span>
                </div>
                <h3 style="font-size: 18px; margin: 0 0 8px 0;">LangChain &amp; Google Gemini API</h3>
                <p style="font-size: 14px; margin: 0;">
                    Segmentação contextual inteligente preservando estrutura documental e geração controlada
                    com aterramento estrito (strict grounding) e abstinência nos fragmentos recuperados.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="ka-card" style="height: 100%;">
                <div style="margin-bottom: 12px;">
                    <span class="ka-badge ka-badge-violet">Rastreabilidade Total</span>
                </div>
                <h3 style="font-size: 18px; margin: 0 0 8px 0;">Citações &amp; Auditoria</h3>
                <p style="font-size: 14px; margin: 0;">
                    Cada resposta do assistente apresenta as referências exatas de documento, número do fragmento
                    e grau de similaridade semântica no painel lateral.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 2. DASHBOARD (INDICADORES, DOCUMENTOS RECENTES, STATUS DE PROCESSAMENTO)
# =============================================================================
def render_dashboard_view(stats: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="ka-page-header">
            <div>
                <div class="ka-page-kicker">VISÃO OPERACIONAL</div>
                <h2 class="ka-page-title">Dashboard de Inteligência Documental</h2>
                <p class="ka-page-subtitle">
                    Monitoramento em tempo real da base documental, fragmentos vetoriais e integridade do pipeline RAG.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Indicadores de documentos e fragmentos
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="ka-metric-card">
                <div class="ka-metric-label">
                    <span>Documentos na Base</span>
                    <span class="ka-badge ka-badge-violet">{stats.get("indexed_documents", 0)} ativos</span>
                </div>
                <div class="ka-metric-value">{stats.get("total_documents", 0)}</div>
                <div class="ka-metric-sub">Volume: {format_bytes(stats.get("total_bytes", 0))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        emb_dim = stats.get("pipeline", {}).get("embedding_dimension", 768)
        st.markdown(
            f"""
            <div class="ka-metric-card">
                <div class="ka-metric-label">
                    <span>Fragmentos Indexados</span>
                    <span class="ka-badge ka-badge-neutral">{emb_dim} dims</span>
                </div>
                <div class="ka-metric-value">{stats.get("total_chunks", 0)}</div>
                <div class="ka-metric-sub">Janela: {stats.get("pipeline", {}).get("chunk_size", 900)} caracteres</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="ka-metric-card">
                <div class="ka-metric-label">
                    <span>Histórico de Conversas</span>
                    <span class="ka-badge ka-badge-neutral">RAG Chat</span>
                </div>
                <div class="ka-metric-value">{stats.get("total_conversations", 0)}</div>
                <div class="ka-metric-sub">{stats.get("total_messages", 0)} interações registradas</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        err_count = stats.get("error_documents", 0)
        proc_count = stats.get("processing_documents", 0)
        health_label = "100% Operacional" if err_count == 0 else f"{err_count} com falha"
        st.markdown(
            f"""
            <div class="ka-metric-card">
                <div class="ka-metric-label">
                    <span>Status de Indexação</span>
                    <span class="ka-badge ka-badge-violet">Online</span>
                </div>
                <div class="ka-metric-value" style="font-size: 24px; padding-top: 6px;">{health_label}</div>
                <div class="ka-metric-sub">{proc_count} em fila de processamento</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.8, 1.2])

    # Área de Documentos Recentes
    with left_col:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <h3 style="font-size: 18px; margin: 0;">Documentos Recentes</h3>
                <span style="font-size: 13px; color: var(--color-graphite);">Últimas ingestões no repositório</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        recent_docs = stats.get("recent_documents", [])
        if not recent_docs:
            st.markdown(
                """
                <div class="ka-card-cloud">
                    <p style="margin: 0;">Nenhum documento encontrado. Acesse a Biblioteca Documental para realizar o primeiro upload.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for doc in recent_docs:
                doc_name = html.escape(doc["filename"])
                f_badge = format_badge_html(doc["file_type"])
                s_badge = status_badge_html(doc["status"])
                st.markdown(
                    f"""
                    <div class="ka-card" style="padding: 16px 24px; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
                            <div>
                                <div style="font-size: 15px; font-weight: 600; color: var(--color-studio-slate); margin-bottom: 4px;">
                                    {doc_name}
                                </div>
                                <div style="font-size: 12px; color: var(--color-graphite);">
                                    {doc["chunk_count"]} fragmentos vetoriais · {format_bytes(doc["file_size"])} · Atualizado em {format_iso_date(doc["updated_at"])}
                                </div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                {f_badge}
                                {s_badge}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Gerenciar Biblioteca Documental", key="dash_to_lib", type="secondary", use_container_width=True):
                navigate_to("biblioteca")
        with b_col2:
            if st.button("Consultar no Assistente →", key="dash_to_chat", type="primary", use_container_width=True):
                navigate_to("assistente")

    # Status de Processamento e Infraestrutura
    with right_col:
        pipeline = stats.get("pipeline", {})
        format_breakdown = stats.get("format_breakdown", {})
        formats_html = "".join(
            f'<span class="ka-badge ka-badge-neutral" style="margin-right: 6px;">{html.escape(k)}: {v}</span>'
            for k, v in format_breakdown.items()
        ) or '<span style="font-size: 13px; color: var(--color-graphite);">Sem arquivos</span>'

        st.markdown(
            f"""
            <div style="margin-bottom: 16px;">
                <h3 style="font-size: 18px; margin: 0;">Status de Processamento &amp; RAG</h3>
            </div>
            <div class="ka-card" style="padding: 24px;">
                <div style="margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--color-mist);">
                    <div style="font-size: 12px; font-weight: 600; color: var(--color-graphite); text-transform: uppercase; margin-bottom: 6px;">
                        Camada de Persistência Vetorial
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 14px; font-weight: 600; color: var(--color-studio-slate);">
                            {html.escape(pipeline.get("storage_mode", "Supabase + pgvector"))}
                        </span>
                        <span class="ka-badge ka-badge-violet">Ativo</span>
                    </div>
                </div>
                <div style="margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--color-mist);">
                    <div style="font-size: 12px; font-weight: 600; color: var(--color-graphite); text-transform: uppercase; margin-bottom: 6px;">
                        Modelos Google Gemini &amp; Embeddings
                    </div>
                    <div style="font-size: 13px; color: var(--color-iron); margin-bottom: 4px;">
                        <strong>LLM:</strong> {html.escape(pipeline.get("llm_model", "gemini-2.5-flash"))}
                    </div>
                    <div style="font-size: 13px; color: var(--color-iron);">
                        <strong>Embedding:</strong> {html.escape(pipeline.get("embedding_model", "models/gemini-embedding-001"))} ({pipeline.get("embedding_dimension", 768)}d)
                    </div>
                </div>
                <div style="margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--color-mist);">
                    <div style="font-size: 12px; font-weight: 600; color: var(--color-graphite); text-transform: uppercase; margin-bottom: 6px;">
                        Parâmetros LangChain Splitter
                    </div>
                    <div style="font-size: 13px; color: var(--color-iron);">
                        Chunk Size: <strong>{pipeline.get("chunk_size", 900)}</strong> chars · Overlap: <strong>{pipeline.get("chunk_overlap", 150)}</strong> chars
                    </div>
                </div>
                <div>
                    <div style="font-size: 12px; font-weight: 600; color: var(--color-graphite); text-transform: uppercase; margin-bottom: 8px;">
                        Distribuição por Formato
                    </div>
                    <div>{formats_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 3. BIBLIOTECA DOCUMENTAL (UPLOAD, LISTA, FORMATOS, STATUS, GERENCIAMENTO)
# =============================================================================
def render_library_view() -> None:
    st.markdown(
        """
        <div class="ka-page-header">
            <div>
                <div class="ka-page-kicker">GESTÃO DE CONHECIMENTO</div>
                <h2 class="ka-page-title">Biblioteca Documental</h2>
                <p class="ka-page-subtitle">
                    Envie documentos corporativos para extração, fragmentação semântica via LangChain e indexação vetorial no pgvector.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Área de Upload
    st.markdown(
        """
        <div class="ka-card" style="padding: 24px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div>
                    <span class="ka-badge ka-badge-violet" style="margin-bottom: 8px;">Ingestão Direta no Pipeline</span>
                    <h3 style="font-size: 18px; margin: 4px 0;">Área de Upload Documental</h3>
                    <p style="font-size: 14px; margin: 0;">
                        Formatos suportados: <strong>PDF, DOCX, Markdown (.md) e Texto (.txt)</strong>. O documento é segmentado e vetorizado automaticamente.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    up_col1, up_col2 = st.columns([3, 1])
    with up_col1:
        uploaded_files = st.file_uploader(
            "Selecione um ou mais arquivos documentais (.pdf, .docx, .md, .txt)",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
        )
    with up_col2:
        st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        process_btn = st.button(
            "Processar e Indexar",
            key="btn_upload_index",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

    if process_btn and uploaded_files:
        for uf in uploaded_files:
            with st.spinner(f"Extraindo e indexando '{uf.name}' no pipeline RAG..."):
                try:
                    raw_bytes = uf.getvalue()
                    indexed_doc = api_client.upload_document(
                        filename=uf.name,
                        file_bytes=raw_bytes,
                        content_type=uf.type or "application/octet-stream",
                        token=st.session_state.auth_token,
                    )
                    st.success(
                        f"Documento '{indexed_doc['filename']}' indexado com sucesso ({indexed_doc['chunk_count']} fragmentos gerados)."
                    )
                except Exception as exc:
                    st.error(f"Erro ao processar '{uf.name}': {exc}")
        st.rerun()

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Filtro de busca e lista de documentos
    docs = api_client.list_documents(token=st.session_state.auth_token)

    f_col1, f_col2 = st.columns([2.5, 1])
    with f_col1:
        search_term = st.text_input(
            "Filtrar documentos por nome",
            placeholder="Digite o nome do arquivo para filtrar...",
        )
    with f_col2:
        format_filter = st.selectbox(
            "Filtrar por formato",
            options=["Todos", "PDF", "MD", "TXT", "DOCX"],
        )

    filtered_docs = docs
    if search_term.strip():
        q_lower = search_term.strip().lower()
        filtered_docs = [d for d in filtered_docs if q_lower in d["filename"].lower()]
    if format_filter != "Todos":
        filtered_docs = [d for d in filtered_docs if d["file_type"].upper() == format_filter]

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin: 16px 0;">
            <h3 style="font-size: 18px; margin: 0;">Documentos Indexados ({len(filtered_docs)})</h3>
            <span style="font-size: 13px; color: var(--color-graphite);">Gerenciamento individual de indexação e fragmentos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not filtered_docs:
        st.markdown(
            """
            <div class="ka-card-cloud">
                <p style="margin: 0;">Nenhum documento corresponde aos critérios selecionados.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for doc in filtered_docs:
            doc_id = doc["id"]
            doc_name = html.escape(doc["filename"])
            f_badge = format_badge_html(doc["file_type"])
            s_badge = status_badge_html(doc["status"])

            st.markdown(
                f"""
                <div class="ka-card" style="padding: 20px 24px; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                                <span style="font-size: 16px; font-weight: 600; color: var(--color-studio-slate);">
                                    {doc_name}
                                </span>
                                {f_badge}
                                {s_badge}
                            </div>
                            <div style="font-size: 13px; color: var(--color-iron);">
                                <strong>{doc["chunk_count"]}</strong> fragmentos vetoriais · Tamanho: <strong>{format_bytes(doc["file_size"])}</strong> · Indexado em {format_iso_date(doc["updated_at"])}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ações de gerenciamento conectadas às rotas FastAPI reais
            act1, act2, act3, act4 = st.columns([1.2, 1.2, 1.0, 1.0])
            with act1:
                is_inspecting = st.session_state.inspected_doc_id == doc_id
                btn_label = "Ocultar Fragmentos" if is_inspecting else "Inspecionar Fragmentos"
                if st.button(btn_label, key=f"inspect_{doc_id}", type="secondary", use_container_width=True):
                    st.session_state.inspected_doc_id = None if is_inspecting else doc_id
                    st.rerun()
            with act2:
                if st.button(
                    "Consultar no Assistente",
                    key=f"ask_doc_{doc_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.assistant_doc_filter = [doc_id]
                    navigate_to("assistente")
            with act3:
                if st.button("Reindexar", key=f"reindex_{doc_id}", type="secondary", use_container_width=True):
                    with st.spinner(f"Reindexando '{doc['filename']}'..."):
                        api_client.reindex_document(doc_id, token=st.session_state.auth_token)
                    st.rerun()
            with act4:
                if st.button("Excluir", key=f"delete_{doc_id}", type="secondary", use_container_width=True):
                    api_client.delete_document(doc_id, token=st.session_state.auth_token)
                    if st.session_state.inspected_doc_id == doc_id:
                        st.session_state.inspected_doc_id = None
                    st.rerun()

            # Painel de inspeção dos fragmentos vetoriais reais do documento
            if st.session_state.inspected_doc_id == doc_id:
                chunks = api_client.get_document_chunks(doc_id, token=st.session_state.auth_token)
                st.markdown(
                    f"""
                    <div class="ka-card-cloud" style="margin-top: 8px; margin-bottom: 24px;">
                        <div style="font-size: 13px; font-weight: 600; color: var(--color-iris); margin-bottom: 12px;">
                            Inspeção de Fragmentos Indexados — {doc_name} ({len(chunks)} trechos)
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                for ch in chunks:
                    dim_val = ch.get("metadata", {}).get("embedding_dimension", 768)
                    st.markdown(
                        f"""
                        <div class="ka-citation-card" style="background-color: var(--color-white);">
                            <div class="ka-citation-header">
                                <span class="ka-citation-title">Trecho #{ch["chunk_index"]}</span>
                                <span class="ka-badge ka-badge-neutral">{ch["token_count"]} palavras · {dim_val}d vector</span>
                            </div>
                            <div class="ka-citation-excerpt">{html.escape(ch["content"])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)


# =============================================================================
# 4. ASSISTENTE (CHAT LIMPO, RESPOSTAS ESTRUTURADAS, CITAÇÕES, REFERÊNCIAS, HISTÓRICO)
# =============================================================================
def render_assistant_view() -> None:
    st.markdown(
        """
        <div class="ka-page-header">
            <div>
                <div class="ka-page-kicker">CONSULTA SEMÂNTICA &amp; SÍNTESE</div>
                <h2 class="ka-page-title">Assistente de Conhecimento Documental</h2>
                <p class="ka-page-subtitle">
                    Faça perguntas técnicas sobre sua biblioteca documental. Todas as respostas são fundamentadas e acompanhadas das citações originais.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    conversations = api_client.list_conversations(token=st.session_state.auth_token)
    documents = api_client.list_documents(token=st.session_state.auth_token)
    indexed_docs = [d for d in documents if d["status"] == "indexed"]

    # Selecionar automaticamente a conversa mais recente se nenhuma estiver ativa
    if st.session_state.active_conversation_id is None and conversations:
        st.session_state.active_conversation_id = conversations[0]["id"]

    # Layout em 3 colunas:
    # 1. Histórico de Conversas & Escopo Documental
    # 2. Interface de Chat Limpa + Campo de Pergunta + Respostas Estruturadas
    # 3. Painel de Referências Documentais (Citações das Fontes)
    col_hist, col_chat, col_refs = st.columns([1.05, 2.25, 1.3])

    # -------------------------------------------------------------------------
    # COLUNA 1: HISTÓRICO DE CONVERSAS E FILTRO DE ESCOPO
    # -------------------------------------------------------------------------
    with col_hist:
        st.markdown(
            """
            <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--color-graphite); margin-bottom: 12px;">
                Histórico de Conversas
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("+ Nova Conversa", key="btn_new_conv", type="primary", use_container_width=True):
            created = api_client.create_conversation(
                title="Nova Consulta Documental",
                token=st.session_state.auth_token,
            )
            st.session_state.active_conversation_id = created["id"]
            st.session_state.last_citations = []
            st.rerun()

        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

        if not conversations:
            st.markdown(
                """
                <div class="ka-card-cloud" style="padding: 16px;">
                    <p style="font-size: 13px; margin: 0;">Nenhuma conversa anterior. Envie uma pergunta para iniciar.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for conv in conversations:
                c_id = conv["id"]
                is_current = c_id == st.session_state.active_conversation_id
                title_short = conv["title"]
                if len(title_short) > 30:
                    title_short = title_short[:30] + "..."
                if st.button(
                    f"{title_short} ({conv['message_count']})",
                    key=f"sel_conv_{c_id}",
                    type="primary" if is_current else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.active_conversation_id = c_id
                    st.session_state.last_citations = []
                    st.rerun()

            if st.session_state.active_conversation_id:
                if st.button(
                    "Excluir Conversa Atual",
                    key="btn_del_conv",
                    type="secondary",
                    use_container_width=True,
                ):
                    api_client.delete_conversation(
                        st.session_state.active_conversation_id,
                        token=st.session_state.auth_token,
                    )
                    st.session_state.active_conversation_id = None
                    st.session_state.last_citations = []
                    st.rerun()

        st.markdown(
            '<div style="margin: 20px 0; border-top: 1px solid var(--color-mist);"></div>',
            unsafe_allow_html=True,
        )

        # Filtro opcional de escopo documental
        doc_options = {d["filename"]: d["id"] for d in indexed_docs}
        default_names = [
            d["filename"]
            for d in indexed_docs
            if d["id"] in st.session_state.assistant_doc_filter
        ]
        selected_names = st.multiselect(
            "Escopo Documental (opcional)",
            options=list(doc_options.keys()),
            default=default_names,
            placeholder="Todos os documentos indexados",
        )
        selected_doc_ids = [doc_options[name] for name in selected_names]
        st.session_state.assistant_doc_filter = selected_doc_ids

        top_k_val = st.slider(
            "Fragmentos recuperados (Top-K)",
            min_value=2,
            max_value=8,
            value=4,
        )

    # -------------------------------------------------------------------------
    # COLUNA 2: INTERFACE DE CHAT LIMPA, PERGUNTA E RESPOSTAS ESTRUTURADAS
    # -------------------------------------------------------------------------
    with col_chat:
        messages: list[dict[str, Any]] = []
        if st.session_state.active_conversation_id:
            try:
                messages = api_client.get_conversation_messages(
                    st.session_state.active_conversation_id,
                    token=st.session_state.auth_token,
                )
            except Exception:
                messages = []

        # Atualizar painel de referências com as citações da última resposta da conversa
        active_citations = st.session_state.last_citations
        if not active_citations and messages:
            for msg in reversed(messages):
                if msg["role"] == "assistant" and msg.get("citations"):
                    active_citations = msg["citations"]
                    break

        # Sugestões rápidas de perguntas conectadas ao pipeline RAG
        st.markdown(
            """
            <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--color-graphite); margin-bottom: 12px;">
                Consultas Rápidas Sugeridas
            </div>
            """,
            unsafe_allow_html=True,
        )
        sug_cols = st.columns(3)
        suggested_prompts = [
            (
                "Arquitetura do Pipeline RAG",
                "Como funciona a arquitetura do pipeline RAG, o chunking no LangChain e a vetorização no pgvector?",
            ),
            (
                "Segurança e Governança JWT",
                "Quais são as regras de segurança, autenticação JWT e prevenção de alucinações exigidas pela política?",
            ),
            (
                "Operação do Supabase pgvector",
                "Como opera a função RPC match_document_chunks no Supabase pgvector e como reindexar documentos?",
            ),
        ]
        triggered_question: Optional[str] = None
        for idx, (label, full_q) in enumerate(suggested_prompts):
            with sug_cols[idx]:
                if st.button(label, key=f"sug_q_{idx}", type="secondary", use_container_width=True):
                    triggered_question = full_q

        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

        # Formulário de Pergunta com campo branco (border-radius 6px) e foco violeta
        with st.form("rag_question_form", clear_on_submit=True):
            user_question = st.text_area(
                "Sua pergunta sobre os documentos indexados",
                placeholder="Digite sua pergunta técnica (ex.: Quais os parâmetros de chunking e dimensão dos embeddings utilizados?)...",
                height=88,
            )
            f_btn_col1, f_btn_col2 = st.columns([2.2, 1.2])
            with f_btn_col1:
                st.markdown(
                    '<div style="font-size: 12px; color: var(--color-graphite); padding-top: 8px;">Respostas fundamentadas exclusivamente nos documentos indexados com citação obrigatória de fonte.</div>',
                    unsafe_allow_html=True,
                )
            with f_btn_col2:
                submit_q = st.form_submit_button(
                    "Consultar Base RAG →",
                    use_container_width=True,
                )

        if submit_q and user_question.strip():
            triggered_question = user_question.strip()

        if triggered_question:
            with st.spinner("Consultando vetores no pgvector e sintetizando resposta estruturada..."):
                try:
                    query_res = api_client.query_assistant(
                        question=triggered_question,
                        conversation_id=st.session_state.active_conversation_id,
                        top_k=top_k_val,
                        document_ids=selected_doc_ids if selected_doc_ids else None,
                        token=st.session_state.auth_token,
                    )
                    st.session_state.active_conversation_id = query_res["conversation_id"]
                    st.session_state.last_citations = query_res["citations"]
                    st.session_state.last_pipeline_meta = query_res.get("pipeline_metadata", {})
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erro ao processar consulta no pipeline RAG: {exc}")

        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

        # Renderização limpa do histórico de mensagens da conversa ativa
        if not messages:
            st.markdown(
                """
                <div class="ka-card" style="padding: 32px; text-align: center;">
                    <div style="margin-bottom: 12px;">
                        <span class="ka-badge ka-badge-violet">Assistente RAG Pronto</span>
                    </div>
                    <h3 style="font-size: 20px; margin: 0 0 8px 0;">Inicie uma Consulta Documental</h3>
                    <p style="font-size: 14px; max-width: 520px; margin: 0 auto;">
                        Utilize o campo acima ou selecione uma das consultas sugeridas para recuperar fragmentos vetoriais
                        da Biblioteca Documental e receber respostas estruturadas com citações verificáveis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in messages:
                role = msg["role"]
                created_fmt = format_iso_date(msg["created_at"])
                if role == "user":
                    st.markdown(
                        f"""
                        <div class="ka-chat-message-user">
                            <div class="ka-chat-role" style="color: var(--color-iron);">
                                <span>PERGUNTA DO USUÁRIO</span>
                                <span style="font-weight: 400; color: var(--color-graphite);">{created_fmt}</span>
                            </div>
                            <div style="font-size: 15px; font-weight: 500; color: var(--color-studio-slate);">
                                {html.escape(msg["content"])}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    msg_citations = msg.get("citations") or []
                    citation_pills = "".join(
                        f'<span class="ka-badge ka-badge-violet" style="margin-right: 6px; margin-top: 6px;">'
                        f'{html.escape(c["filename"])} · Trecho #{c["chunk_index"]} ({int(c["similarity"] * 100)}%)</span>'
                        for c in msg_citations
                    )
                    st.markdown(
                        f"""
                        <div class="ka-chat-message-assistant">
                            <div class="ka-chat-role" style="color: var(--color-iris);">
                                <span>◈ RESPOSTA ESTRUTURADA · KNOWLEDGE ASSISTANT</span>
                                <span style="font-weight: 400; color: var(--color-graphite);">{created_fmt}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg["content"])
                    if citation_pills:
                        st.markdown(
                            f"""
                            <div style="margin-top: 8px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--color-mist);">
                                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--color-graphite); margin-bottom: 4px;">
                                    Fontes Citadas nesta Resposta:
                                </div>
                                <div>{citation_pills}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # -------------------------------------------------------------------------
    # COLUNA 3: PAINEL DE REFERÊNCIAS DOCUMENTAIS
    # -------------------------------------------------------------------------
    with col_refs:
        st.markdown(
            """
            <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--color-graphite); margin-bottom: 12px;">
                Painel de Referências Documentais
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not active_citations:
            st.markdown(
                """
                <div class="ka-card-cloud" style="padding: 20px;">
                    <div style="font-size: 14px; font-weight: 600; color: var(--color-studio-slate); margin-bottom: 6px;">
                        Aguardando Consulta
                    </div>
                    <p style="font-size: 13px; margin: 0;">
                        Os fragmentos recuperados via busca semântica no pgvector serão exibidos aqui com seu score de similaridade e trecho original.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="margin-bottom: 12px;">
                    <span class="ka-badge ka-badge-violet">{len(active_citations)} fragmentos recuperados</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for idx, cit in enumerate(active_citations, start=1):
                sim_pct = int(float(cit.get("similarity", 0.0)) * 100)
                fname = html.escape(cit.get("filename", "Documento"))
                ftype = html.escape(cit.get("file_type", "DOC"))
                c_idx = cit.get("chunk_index", idx)
                excerpt = html.escape(cit.get("excerpt", ""))

                st.markdown(
                    f"""
                    <div class="ka-citation-card">
                        <div class="ka-citation-header">
                            <span class="ka-badge ka-badge-violet">Trecho #{c_idx} · {sim_pct}% relevância</span>
                            <span class="ka-badge ka-badge-neutral">{ftype}</span>
                        </div>
                        <div class="ka-citation-title" style="margin-bottom: 8px;">
                            {fname}
                        </div>
                        <div class="ka-citation-excerpt">
                            {excerpt}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# ENTRYPOINT PRINCIPAL DO FRONTEND STREAMLIT
# =============================================================================
def main() -> None:
    load_specify_css()
    init_session_state()

    stats = api_client.get_dashboard_stats(token=st.session_state.auth_token)

    render_sidebar(stats)
    render_top_navigation()

    current_view = st.session_state.active_view
    if current_view == "inicio":
        render_home_view(stats)
    elif current_view == "dashboard":
        render_dashboard_view(stats)
    elif current_view == "biblioteca":
        render_library_view()
    elif current_view == "assistente":
        render_assistant_view()
    else:
        render_home_view(stats)


if __name__ == "__main__":
    main()
