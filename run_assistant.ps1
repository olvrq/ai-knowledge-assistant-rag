# =============================================================================
# AI KNOWLEDGE ASSISTANT — SCRIPT DE INICIALIZAÇÃO RÁPIDA
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==> Verificando ambiente virtual Python..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
$VenvUvicorn = Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe"

Write-Host "==> Instalando/verificando dependências (FastAPI, Streamlit, Supabase, LangChain, Google Gemini)..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r frontend/requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> Arquivo .env criado a partir de .env.example." -ForegroundColor Yellow
}

Write-Host "==> Iniciando Backend FastAPI na porta 8000..." -ForegroundColor Green
Start-Process -FilePath $VenvUvicorn -ArgumentList "backend.main:app --host 127.0.0.1 --port 8000" -NoNewWindow

Write-Host "==> Iniciando Frontend Streamlit (Design System Specify)..." -ForegroundColor Green
& $VenvStreamlit run frontend/app.py
