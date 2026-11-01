"""
Entrypoint Serverless da Vercel para o Backend FastAPI do AI Knowledge Assistant.
Garante que o diretório raiz do projeto esteja no sys.path antes de importar `backend.main:app`.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app  # noqa: E402, F401
