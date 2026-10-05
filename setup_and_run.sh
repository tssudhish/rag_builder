#!/usr/bin/env bash
# ==============================================================================
# RAG Knowledge Graph Builder - Automated Setup & Launch Script
# Clones repo, sets up local Python virtual environment (.venv), installs dependencies,
# pulls Ollama models (if available), and starts the FastAPI/Web application.
# ==============================================================================

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tssudhish/rag_builder.git}"
TARGET_DIR="${1:-rag_builder}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
VENV_DIR=".venv"

echo "=========================================================="
echo "  RAG Knowledge Graph Builder - Setup & Run"
echo "=========================================================="

# 1. Clone Repository if not already in project directory
if [ ! -f "pyproject.toml" ]; then
    if [ -d "$TARGET_DIR" ]; then
        echo "--> Directory '$TARGET_DIR' already exists. Navigating into it..."
        cd "$TARGET_DIR"
    else
        echo "--> Cloning repository from $REPO_URL into $TARGET_DIR..."
        git clone "$REPO_URL" "$TARGET_DIR"
        cd "$TARGET_DIR"
    fi
else
    echo "--> Already inside project directory ($(pwd)). Proceeding..."
fi

# 2. Locate a real, functioning Python executable (skips WindowsApps store shims)
echo "--> Locating valid Python 3.9+ installation..."

test_python() {
    local cmd="$1"
    [ -z "$cmd" ] && return 1

    # Ignore WindowsApps dummy aliases
    if echo "$cmd" | grep -qi "WindowsApps"; then
        return 1
    fi

    # Check if executable exists or command exists
    local resolved_path=""
    if [ -f "$cmd" ]; then
        resolved_path="$cmd"
    elif command -v "$cmd" &>/dev/null; then
        resolved_path=$(command -v "$cmd")
    else
        return 1
    fi

    if echo "$resolved_path" | grep -qi "WindowsApps"; then
        return 1
    fi

    # Verify that this python binary can execute code and is >= 3.8
    if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' &>/dev/null; then
        return 0
    fi
    return 1
}

find_python() {
    # Check explicit env override first
    if [ -n "${PYTHON_BIN:-}" ] && test_python "${PYTHON_BIN}"; then
        echo "${PYTHON_BIN}"
        return 0
    fi

    local current_user="${USER:-${USERNAME:-}}"
    local candidates=(
        "python"
        "python3"
        "py"
        "/d/Users/${current_user}/miniconda3/python.exe"
        "/c/Users/${current_user}/miniconda3/python.exe"
        "/c/Users/${current_user}/anaconda3/python.exe"
        "/d/miniconda3/python.exe"
        "/c/miniconda3/python.exe"
        "D:/Users/${current_user}/miniconda3/python.exe"
        "C:/Users/${current_user}/miniconda3/python.exe"
        "C:/Users/${current_user}/AppData/Local/Programs/Python/Python312/python.exe"
        "C:/Users/${current_user}/AppData/Local/Programs/Python/Python311/python.exe"
        "C:/Users/${current_user}/AppData/Local/Programs/Python/Python310/python.exe"
        "C:/Users/${current_user}/AppData/Local/Programs/Python/Python39/python.exe"
        "C:/Program Files/Python312/python.exe"
        "C:/Program Files/Python311/python.exe"
        "C:/Program Files/Python310/python.exe"
        "C:/Program Files/Python39/python.exe"
    )

    for cand in "${candidates[@]}"; do
        if test_python "$cand"; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

if ! BASE_PYTHON=$(find_python); then
    echo "=========================================================="
    echo "Error: Could not find a working Python 3.9+ binary."
    echo "If you have Python installed, you can pass its path via:"
    echo "  export PYTHON_BIN=/path/to/python.exe"
    echo "  bash setup_and_run.sh"
    echo "=========================================================="
    exit 1
fi

echo "    Using base Python: $BASE_PYTHON"
BASE_VERSION=$("$BASE_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "    Base Python version: $BASE_VERSION"

# 3. Create Local Python Virtual Environment (.venv)
if [ ! -d "$VENV_DIR" ]; then
    echo "--> Creating local virtual environment in $VENV_DIR..."
    "$BASE_PYTHON" -m venv "$VENV_DIR"
else
    echo "--> Local virtual environment '$VENV_DIR' already exists."
fi

# 4. Activate the Local Virtual Environment
echo "--> Activating local virtual environment ($VENV_DIR)..."
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Windows Git Bash / MSYS2
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    # POSIX / Linux / macOS / WSL
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment activation script not found in $VENV_DIR."
    exit 1
fi

echo "    Active Python is now: $(which python)"

# 5. Upgrade pip & Install Dependencies inside .venv
echo "--> Upgrading pip inside local venv..."
python -m pip install --upgrade pip

echo "--> Installing rag_builder in editable mode..."
python -m pip install -e .

echo "--> Installing API and vector dependencies into local venv..."
python -m pip install fastapi uvicorn python-multipart numpy neo4j

# 6. Ensure required runtime directories exist
echo "--> Ensuring data directories exist..."
mkdir -p data/uploads

# 7. Check & prepare Ollama models (optional)
echo "--> Checking Ollama status..."
if command -v ollama &>/dev/null; then
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "    Ollama server is active. Ensuring models are pulled..."
        ollama pull sciphi-triplex || echo "    [Notice] Could not pull sciphi-triplex automatically."
        ollama pull gemma2 || echo "    [Notice] Could not pull gemma2 automatically."
    else
        echo "    [Warning] Ollama is installed but not running at http://localhost:11434."
        echo "    Start Ollama ('ollama serve') in another terminal for LLM triplet extraction."
    fi
else
    echo "    [Warning] 'ollama' CLI not found. Install from https://ollama.ai/ if using local LLM."
fi

# 8. Launch Application with local venv python
echo "=========================================================="
echo "  Setup complete! Running inside local venv: $VENV_DIR"
echo "  - Web Visualizer: http://localhost:${APP_PORT}/web/index.html"
echo "  - Swagger Docs:   http://localhost:${APP_PORT}/docs"
echo "  - CLI Tool:       rag-builder --help"
echo "=========================================================="

exec python -m uvicorn rag_builder.api.main:app --host "$APP_HOST" --port "$APP_PORT" --reload
