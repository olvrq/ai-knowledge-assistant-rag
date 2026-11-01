# DIREÇÃO VISUAL OFICIAL — AI KNOWLEDGE ASSISTANT

Este documento define o Design System e as regras visuais estritas do **AI Knowledge Assistant**, reinterpretando a linguagem estética SaaS tecnológica, moderna e sofisticada da referência **Specify** para um produto de inteligência artificial especializado em conhecimento documental e RAG.

---

## 1. Identidade Visual

- **Seções de apresentação (Hero)** com fundo escuro (`--color-obsidian` / `--color-studio-slate`).
- **Headline** com gradiente violeta exclusivo para o título de destaque do hero.
- **Áreas internas de aplicação** predominantemente brancas (`--color-white`).
- **Superfícies secundárias** em cinza muito claro (`--color-cloud`).
- **Elementos de destaque** em violeta (`--color-iris`, `--color-soft-iris`, `--color-lilac-wash`).
- **Cards arredondados** (`16px`) com bordas sutis (`--color-mist`) e sombras discretas.
- **Tipografia Inter** em toda a aplicação.

---

## 2. Paleta Oficial (CSS Custom Properties)

```css
:root {
  --color-iris: #624de3;
  --color-soft-iris: #8d4af7;
  --color-studio-slate: #1a1d1e;
  --color-obsidian: #151718;
  --color-white: #ffffff;
  --color-cloud: #f6f7f9;
  --color-mist: #ebedef;
  --color-graphite: #8d8e8f;
  --color-iron: #5f6162;
  --color-noir: #000000;
  --color-lilac-wash: #f1eefe;
}
```

---

## 3. Tipografia

- **Fonte**: `Inter` (`400`, `500`, `600`, `700`)
- **Headlines**: `48px` a `64px`, peso `700`, `letter-spacing: -0.03em`, gradiente violeta apenas nos títulos de destaque do hero.
- **Corpo**: `14px` a `16px`, `line-height: 1.6`, contraste elevado e excelente legibilidade.

---

## 4. Componentes

- **Cards**: Fundo `#ffffff`, `border-radius: 16px`, borda `1px solid #ebedef`, sombra discreta.
- **Botões**: `border-radius: 40px` (formato pílula).
  - Principal: Fundo `#1a1d1e` e texto `#ffffff`.
  - Secundário: Fundo `#ffffff` e borda `1px solid #ebedef`.
  - Violeta reservado para destaques e elementos de identidade.
- **Campos**: `border-radius: 6px`, fundo `#ffffff`, borda `1px solid #ebedef`, foco em `#624de3` com anel `#f1eefe`.

---

## 5. Regras Estritas

1. Não utilizar gradientes como fundo geral da aplicação.
2. Não utilizar glassmorphism (`backdrop-filter: blur`).
3. Não adicionar fotografias ou ilustrações decorativas genéricas.
4. Não utilizar cores fora da paleta oficial.
5. Não utilizar botões quadrados.
6. Manter grade de espaçamento múltipla de `8px`.
7. Preservar integralmente a pilha funcional: **FastAPI, Streamlit, Supabase, pgvector (768d), LangChain, Google Gemini API (`gemini-2.5-flash` & `models/gemini-embedding-001`), Autenticação, Segurança e Pipeline RAG**.
