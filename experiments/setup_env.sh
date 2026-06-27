# Sourced by the run scripts to stand up a uv virtualenv on a fresh clone.
# Mirrors the nanochat task-script convention: install uv if missing, create
# .venv, install deps, activate. Set SKIP_ENV=1 (or already be inside a
# VIRTUAL_ENV) to skip this and use whatever python is active.
#
# Usage (from a run script, after cd-ing to the repo root):
#   source experiments/setup_env.sh

if [ -n "${SKIP_ENV:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "env: using active python ($(command -v python)) — skipping uv setup"
else
    if ! command -v uv &> /dev/null; then
        echo "env: installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    [ -d ".venv" ] || uv venv
    echo "env: uv pip install -r requirements.txt"
    uv pip install -r requirements.txt
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
echo "env: python = $(command -v python)   torch = $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
