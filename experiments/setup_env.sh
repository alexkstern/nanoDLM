# Sourced by the run scripts to provide a working python with torch/numpy/tiktoken.
# Preference order:
#   1. SKIP_ENV=1 or already inside a VIRTUAL_ENV -> use the active python.
#   2. torch+numpy already importable (Colab/Kaggle/A100 image with preinstalled,
#      usually GPU, torch) -> use it as-is. Do NOT build a fresh uv venv, which
#      would shadow it with a slow CPU-only reinstall. Just add tiktoken if missing.
#   3. otherwise (bare box) -> create a uv .venv and install requirements.txt INTO it.
#
# Set SKIP_ENV=1 to force option 1.

_ensure_tiktoken() { python -c "import tiktoken" 2>/dev/null || python -m pip install -q tiktoken || true; }

if [ -n "${SKIP_ENV:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "env: using active python ($(command -v python)) — SKIP_ENV/VIRTUAL_ENV set"
    _ensure_tiktoken
elif python -c "import torch, numpy" 2>/dev/null; then
    echo "env: torch+numpy already present in $(command -v python) — using it (no uv venv)"
    _ensure_tiktoken
else
    if ! command -v uv &> /dev/null; then
        echo "env: installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    [ -d ".venv" ] || uv venv
    # Activate FIRST so uv targets the venv, and pass --python explicitly so a
    # stray UV_SYSTEM_PYTHON can't redirect the install to the system interpreter.
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "env: uv pip install -r requirements.txt (into $VIRTUAL_ENV)"
    uv pip install --python "$VIRTUAL_ENV/bin/python" -r requirements.txt
fi
echo "env: python = $(command -v python)   torch = $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
