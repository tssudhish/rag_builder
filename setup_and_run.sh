#!/usr/bin/env bash
# ==============================================================================
# RAG Knowledge Graph Builder - Automated Setup & Launch Script
# Clones repo, sets up Python virtual environment, installs dependencies,
# pulls Ollama models (if available), and starts the FastAPI/Web application.
# ==============================================================================

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tssudhish/rag_builder.git}"
TARGET_DIR="${1:-rag_builder}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"

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

# 2. Check Python installation
echo "--> Checking Python version..."
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3.9+ is required but neither python3 nor python was found."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "    Found Python version: $PYTHON_VERSION"

# 3. Create Virtual Environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "--> Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo "--> Virtual environment '$VENV_DIR' already exists."
fi

# 4. Activate Virtual Environment (Supports Unix, macOS, WSL, and Git Bash on Windows)
echo "--> Activating virtual environment..."
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
else
    echo "Error: Virtual environment activation script not found in $VENV_DIR."
    exit 1
fi

# 5. Upgrade pip & Install Dependencies
echo "--> Upgrading pip..."
pip install --upgrade pip

echo "--> Installing rag_builder in editable mode with dependencies..."
pip install -e .

echo "--> Installing API and vector dependencies..."
pip install fastapi uvicorn python-multipart numpy neo4j

# 6. Ensure required runtime directories exist
echo "--> Ensuring data directories exist..."
mkdir -p data/uploads

# 7. Check & prepare Ollama models (optional but recommended)
echo "--> Checking Ollama status..."
if command -v ollama &>/dev/null; then
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "    Ollama server is active. Ensuring models are pulled..."
        ollama pull sciphi-triplex || echo "    [Notice] Could not pull sciphi-triplex automatically."
        ollama pull gemma2 || echo "    [Notice] Could not pull gemma2 automatically."
    else
        echo "    [Warning] Ollama is installed but not running at http://localhost:11434."
        echo "    Please run 'ollama serve' in another terminal for LLM triplet extraction."
    fi
else
    echo "    [Warning] 'ollama' CLI not found. Please install Ollama from https://ollama.ai/"
fi

# 8. Launch Application
echo "=========================================================="
echo "  Setup complete! Starting RAG Knowledge Graph Builder..."
echo "  - Web Visualizer: http://localhost:${APP_PORT}/web/index.html"
echo "  - Swagger Docs:   http://localhost:${APP_PORT}/docs"
echo "  - CLI Tool:       rag-builder --help"
echo "=========================================================="

exec uvicorn rag_builder.api.main:app --host "$APP_HOST" --port "$APP_PORT" --reload
