#!/usr/bin/env python3
"""
Project Manager Orchestrator for RAG Knowledge Graph Builder
Pairs Project Manager AI with local Ollama agents:
  - 'coder': Primary implementation agent (OpenCode + Ollama)
  - 'reviewer': Local code reviewer (Ollama)
  - 'tester': Local test setup and verification agent (Ollama)
  - 'triplet_extractor': Specialist triplet extractor (Ollama Triplex)

Minimizes Gemini tokens by delegating all heavy coding, reviewing, and testing to local Ollama models.
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import urllib.request
import urllib.error
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


WORKSPACE = Path(__file__).resolve().parent
STATE_FILE = WORKSPACE / ".pm_state.json"
ROADMAP_FILE = WORKSPACE / "ROADMAP.md"
ARCHITECTURE_FILE = WORKSPACE / "ARCHITECTURE.md"

def get_python_bin():
    """Resolves active Python executable in an environment-agnostic, portable manner."""
    if os.environ.get("PYTHON_BIN") and Path(os.environ["PYTHON_BIN"]).exists():
        return os.environ["PYTHON_BIN"]
    if Path(sys.executable).exists() and "WindowsApps" not in sys.executable:
        return sys.executable
    if os.environ.get("CONDA_PREFIX"):
        conda_py = Path(os.environ["CONDA_PREFIX"]) / "python.exe"
        if conda_py.exists():
            return str(conda_py)
    for candidate in [
        r"D:\Users\tssud\miniconda3\python.exe",
        Path.home() / "miniconda3" / "python.exe",
        Path.home() / "anaconda3" / "python.exe",
    ]:
        if Path(candidate).exists():
            return str(candidate)
    for name in ["python3", "python"]:
        found = shutil.which(name)
        if found and "WindowsApps" not in found:
            return found
    return sys.executable

def get_opencode_bin():
    """Resolves opencode CLI binary portably across operating systems."""
    if os.environ.get("OPENCODE_EXE"):
        return os.environ["OPENCODE_EXE"]
    for name in ["opencode", "opencode.cmd", "opencode.exe"]:
        found = shutil.which(name)
        if found:
            return found
    appdata = os.environ.get("APPDATA", "")
    candidates = [
        r"D:\Users\tssud\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe",
        r"D:\Users\tssud\AppData\Roaming\npm\opencode.cmd",
        Path(appdata) / "npm" / "node_modules" / "opencode-ai" / "bin" / "opencode.exe",
        Path(appdata) / "npm" / "opencode.cmd",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return "opencode"

def get_ollama_bin():
    """Resolves ollama CLI binary portably."""
    if os.environ.get("OLLAMA_EXE"):
        return os.environ["OLLAMA_EXE"]
    found = shutil.which("ollama")
    if found:
        return found
    localappdata = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(localappdata) / "Programs" / "Ollama" / "ollama.exe",
        r"C:\Users\tssud\AppData\Local\Programs\Ollama\ollama.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return "ollama"

OPENCODE_EXE = get_opencode_bin()
OLLAMA_EXE = get_ollama_bin()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:1.5b")
TRIPLEX_MODEL = "sciphi/triplex:latest"

TASKS = {
    # Phase 1: Core Extraction Engine
    "task_1_1": {
        "phase": 1,
        "title": "Document Loaders & Text Splitter",
        "description": "Implement document loaders for PDF, TXT, MD, HTML and context-aware text chunking with metadata.",
        "target_files": [
            "rag_builder/__init__.py",
            "rag_builder/ingestion/__init__.py",
            "rag_builder/ingestion/loaders.py",
            "rag_builder/ingestion/splitter.py"
        ],
        "acceptance_criteria": [
            "Loads TXT, Markdown, and PDF files cleanly with error handling",
            "Splits text into chunks preserving sentence boundaries and metadata (source, chunk_id, char_offset)",
            "Exports clean Document and Chunk dataclasses"
        ]
    },
    "task_1_2": {
        "phase": 1,
        "title": "LLM Triplet Extraction Engine",
        "description": "Develop entity and relationship extraction logic interfacing with Ollama (SciPhi Triplex or Gemma).",
        "target_files": [
            "rag_builder/extraction/__init__.py",
            "rag_builder/extraction/triplets.py",
            "rag_builder/extraction/ollama_client.py"
        ],
        "acceptance_criteria": [
            "OllamaClient can connect to local Ollama (http://localhost:11434)",
            "Extracts (Subject, Predicate, Object) triplets from text chunks",
            "Robust JSON parsing handling markdown fenced code blocks and malformed LLM outputs"
        ]
    },
    "task_1_3": {
        "phase": 1,
        "title": "Schema Alignment & Normalization",
        "description": "Normalize entity names (casing, canonical aliases) and relation types to prevent duplicate nodes.",
        "target_files": [
            "rag_builder/extraction/schema.py",
            "rag_builder/extraction/normalizer.py"
        ],
        "acceptance_criteria": [
            "EntityNormalizer canonicalizes entity mentions across multiple chunks",
            "RelationNormalizer maps synonymous verbs to standardized relation types",
            "Validated Triplet data structures with type hints and confidence scores"
        ]
    },
    "task_1_4": {
        "phase": 1,
        "title": "Phase 1 Validation & Test Harness",
        "description": "Create sample documents and comprehensive unit tests validating the ingestion and extraction pipeline.",
        "target_files": [
            "pm_orchestrator.py",
            "verify_ollama.py",
            "tests/__init__.py",
            "tests/test_loaders.py",
            "tests/test_splitter.py",
            "tests/test_extraction.py",
            "tests/test_normalization.py"
        ],
        "acceptance_criteria": [
            "Unit tests covering file loading, text splitting, and mock triplet extraction",
            "Verification script that tests Ollama connectivity and triplet generation on sample text"
        ]
    },

    # Phase 2: Graph Storage & Schema
    "task_2_1": {
        "phase": 2,
        "title": "Graph Database Interface & Schema",
        "description": "Define Graph Database adapter (Neo4j / FalkorDB / in-memory NetworkX fallback) with node/edge schemas.",
        "target_files": [
            "rag_builder/storage/__init__.py",
            "rag_builder/storage/base.py",
            "rag_builder/storage/neo4j_adapter.py",
            "rag_builder/storage/memory_adapter.py",
            "tests/test_storage.py"
        ],
        "acceptance_criteria": [
            "BaseGraphStorage interface defining insert_triplet, query_neighbors, get_stats",
            "In-memory fallback adapter for lightweight local testing without database server",
            "Neo4j adapter supporting Cypher queries and connection pooling"
        ]
    },
    "task_2_2": {
        "phase": 2,
        "title": "Triplet Ingestion Pipeline",
        "description": "Build ingestion pipeline from extracted triplets to graph database with deduplication and vector embeddings.",
        "target_files": [
            "rag_builder/storage/pipeline.py",
            "rag_builder/storage/vector_index.py",
            "tests/test_pipeline.py"
        ],
        "acceptance_criteria": [
            "Batched ingestion of triplets into graph nodes and directed edges",
            "Vector embedding storage on entity nodes for hybrid search"
        ]
    },
    "task_2_3": {
        "phase": 2,
        "title": "Graph Verification & Inspection Queries",
        "description": "Implement graph query tools to inspect node degrees, connected components, and relationship types.",
        "target_files": [
            "rag_builder/storage/inspector.py",
            "tests/test_storage.py",
            "tests/test_inspector.py"
        ],
        "acceptance_criteria": [
            "Diagnostic CLI commands to check graph health and node counts",
            "Automated storage tests"
        ]
    },

    # Phase 3: Backend API & Web Integration
    "task_3_1": {
        "phase": 3,
        "title": "FastAPI Backend & Document Upload Endpoints",
        "description": "Create FastAPI REST server with document upload, processing status, and graph query endpoints.",
        "target_files": [
            "rag_builder/api/__init__.py",
            "rag_builder/api/main.py",
            "rag_builder/api/routes.py",
            "rag_builder/api/state.py",
            "tests/test_api.py"
        ],
        "acceptance_criteria": [
            "POST /api/documents upload endpoint with background processing",
            "GET /api/documents/status endpoint tracking ingestion progress",
            "GET /api/graph/stats endpoint"
        ]
    },
    "task_3_2": {
        "phase": 3,
        "title": "Graph Visualization API & Web UI",
        "description": "Export graph data in Cytoscape/D3 compatible format and provide web visualization interface.",
        "target_files": [
            "rag_builder/api/visualizer.py",
            "rag_builder/web/index.html",
            "rag_builder/web/app.js",
            "rag_builder/web/style.css",
            "tests/test_visualization.py"
        ],
        "acceptance_criteria": [
            "GET /api/graph/nodes-and-edges for UI visualization",
            "Interactive HTML/CSS/JS frontend showing the knowledge graph"
        ]
    },

    # Phase 4: GraphRAG Implementation
    "task_4_1": {
        "phase": 4,
        "title": "Natural Language to Graph Query Translation",
        "description": "Use Ollama LLM to convert user questions into graph retrieval queries (Cypher / neighborhood lookups).",
        "target_files": [
            "rag_builder/rag/__init__.py",
            "rag_builder/rag/query_generator.py",
            "tests/test_query_generator.py"
        ],
        "acceptance_criteria": [
            "Converts natural language queries into target entities and query patterns",
            "Sanitizes query strings against injection"
        ]
    },
    "task_4_2": {
        "phase": 4,
        "title": "Sub-graph Retrieval (k-hop neighborhood)",
        "description": "Fetch target nodes and their k-hop neighbors with edge predicates to form structured context.",
        "target_files": [
            "rag_builder/rag/__init__.py",
            "rag_builder/rag/retriever.py",
            "tests/test_retriever.py"
        ],
        "acceptance_criteria": [
            "Retrieves k-hop subgraphs with relevance scoring from BaseGraphStorage",
            "Formats sub-graph into structured prompt context for LLM",
            "Unit tests covering k-hop retrieval, depth limits, and prompt context formatting"
        ]
    },
    "task_4_3": {
        "phase": 4,
        "title": "Context-Aware Response Generation & Evaluation",
        "description": "Integrate context with LLM response generator and compare answer quality vs vector-only RAG.",
        "target_files": [
            "rag_builder/rag/__init__.py",
            "rag_builder/rag/generator.py",
            "rag_builder/rag/evaluator.py",
            "tests/test_rag.py"
        ],
        "acceptance_criteria": [
            "GraphRAG generator produces grounded responses with entity & relationship citations",
            "Evaluator measures faithfulness and answer completeness vs baseline",
            "Unit tests covering response generation, citation extraction, and evaluation metrics"
        ]
    },

    # Phase 5: Agent Export & DevOps -> v1.0
    "task_5_1": {
        "phase": 5,
        "title": "Agent Tool Definitions (LangChain / Function Calling)",
        "description": "Package the Knowledge Graph as standardized tool definitions for external autonomous agents.",
        "target_files": [
            "rag_builder/agent_tools/__init__.py",
            "rag_builder/agent_tools/langchain_tools.py",
            "tests/test_agent_tools.py"
        ],
        "acceptance_criteria": [
            "Callable tools for query_graph, get_entity_neighbors, search_triplets",
            "Type-safe tool schemas compatible with LangChain / function calling",
            "Unit tests covering all tool execution and schema validation"
        ]
    },
    "task_5_2": {
        "phase": 5,
        "title": "End-to-End DevOps, Test Coverage & Packaging",
        "description": "Finalize CLI entry points, setup.py / pyproject.toml, full test suite, and release v1.0.",
        "target_files": [
            "rag_builder/cli.py",
            "pyproject.toml",
            "tests/test_e2e.py"
        ],
        "acceptance_criteria": [
            "All unit and integration tests pass",
            "CLI tool `rag-builder` works end-to-end",
            "Version 1.0 release tagged and documented"
        ]
    }
}


def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "current_phase": 1,
        "active_task": "task_1_1",
        "tasks": {task_id: {"status": "pending", "reviews": [], "tests": []} for task_id in TASKS}
    }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def query_ollama(prompt, model=DEFAULT_MODEL, system=None):
    """Query local Ollama instance directly via HTTP API (0 Gemini tokens spent)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    if system:
        payload["system"] = system

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except Exception as e:
        return f"[Error connecting to Ollama at {OLLAMA_URL}: {e}]"


def run_opencode_task(task_id, auto_approve=True, feedback=None):
    """Dispatches a task prompt to the local coder agent using opencode."""
    task = TASKS[task_id]
    state = load_state()
    state["active_task"] = task_id
    state["tasks"][task_id]["status"] = "in_progress"
    save_state(state)

    prompt = (
        f"PROJECT MANAGER WORK PACKET: {task_id} - {task['title']}\n\n"
        f"Goal: {task['description']}\n\n"
        f"Target Files:\n" + "\n".join(f"- {f}" for f in task["target_files"]) + "\n\n"
        f"Acceptance Criteria:\n" + "\n".join(f"- {c}" for c in task["acceptance_criteria"]) + "\n\n"
        f"Instructions:\n"
        f"1. Refer to ARCHITECTURE.md and ROADMAP.md for architectural patterns.\n"
        f"2. Write complete, robust, typed Python code. Do not leave placeholder TODOs.\n"
        f"3. Make sure all parent directories exist and files are saved cleanly.\n"
        f"4. If tests or sample data are needed, create them in the appropriate folders.\n"
        f"5. When done, output a summary of files created and key implementation details.\n"
    )

    if feedback:
        prompt += f"\n\nREVIEWER FEEDBACK / REVISION REQUIREMENTS:\n{feedback}\n"

    cmd = [
        OPENCODE_EXE,
        "run",
        prompt,
        "--agent", "coder",
        "--model", f"ollama/{DEFAULT_MODEL}",
    ]
    if auto_approve:
        cmd.append("--auto")

    print(f"\n[Project Manager] Dispatching {task_id} to Ollama coder agent...")
    print(f"[Command] {' '.join(cmd[:5])} ...")

    result = subprocess.run(cmd, cwd=str(WORKSPACE), stdin=subprocess.DEVNULL)
    return result.returncode == 0



def run_local_review(task_id):
    """Invokes local Ollama agent to review generated code (0 Gemini tokens)."""
    task = TASKS[task_id]
    git_diff = subprocess.run(["git", "diff"], cwd=str(WORKSPACE), capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    untracked = subprocess.run(["git", "status", "--short"], cwd=str(WORKSPACE), capture_output=True, text=True, encoding="utf-8", errors="replace").stdout

    files_to_review = list(task["target_files"])
    for line in untracked.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            path_part = parts[-1].replace("\\", "/")
            if path_part.startswith("tests/") and path_part.endswith(".py") and path_part not in files_to_review:
                files_to_review.append(path_part)

    files_content = []
    for fpath_str in files_to_review:
        fpath = WORKSPACE / fpath_str
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8")
                files_content.append(f"=== File: {fpath_str} ===\n{content}\n")
            except Exception as e:
                files_content.append(f"=== File: {fpath_str} (error: {e}) ===\n")

    code_payload = "\n".join(files_content)

    system_prompt = (
        "You are 'reviewer', an expert senior code reviewer. You evaluate code against "
        "production standards: architecture compliance, edge cases, error handling, typing, and readability. "
        "Return your verdict clearly: PASSED, NEEDS_IMPROVEMENT, or FAILED, followed by specific feedback."
    )

    review_prompt = (
        f"Review work for Task {task_id}: {task['title']}.\n"
        f"Acceptance Criteria:\n" + "\n".join(f"- {c}" for c in task["acceptance_criteria"]) + "\n\n"
        f"Current Git Status:\n{untracked}\n\n"
        f"Source Code Submitted:\n{code_payload}\n\n"
        f"Git Diff:\n{git_diff[:4000] if git_diff else 'None'}\n"
    )

    print(f"\n[Project Manager] Invoking local Ollama code reviewer on {task_id}...")
    review = query_ollama(review_prompt, model=DEFAULT_MODEL, system=system_prompt)

    state = load_state()
    state["tasks"][task_id]["reviews"].append(review)
    if "PASSED" in review.upper():
        state["tasks"][task_id]["status"] = "review_passed"
    save_state(state)

    print("\n--- Reviewer Feedback ---")
    print(review)
    print("-------------------------\n")
    return review


def run_local_tests():
    """Runs test suite locally using pytest or unittest."""
    print("\n[Project Manager] Running local test suite...")
    python_bin = get_python_bin()

    res = subprocess.run([python_bin, "-m", "pytest", "tests"], cwd=str(WORKSPACE))
    if res.returncode == 0:
        return True

    result = subprocess.run([python_bin, "-m", "unittest", "discover", "-s", "tests"], cwd=str(WORKSPACE))
    return result.returncode == 0


def sync_git(push=True):
    """Synchronizes repository with origin/main."""
    print("\n[Project Manager] Synchronizing with remote repository...")
    subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=str(WORKSPACE), check=False)
    if push:
        res = subprocess.run(["git", "push", "origin", "main"], cwd=str(WORKSPACE))
        if res.returncode == 0:
            print("[Project Manager] Successfully synchronized with origin/main.")
        else:
            print("[Project Manager] Warning: git push returned non-zero code.")


def cleanup_transient_files():
    """Removes scratch test files created during development runs."""
    for pattern in ["test_doc.*", "test_ingestion.py"]:
        for f in WORKSPACE.glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass


def approve_task(task_id, auto_sync=True):
    """Verifies tests, cleans scratch files, commits with conventional message, and synchronizes to remote."""
    print(f"\n[Project Manager] Verifying deliverables for {task_id} before approval...")
    
    # 1. Verification gate: Tests must pass
    if not run_local_tests():
        print(f"\n❌ [Project Manager] ABORT: Test suite failed! Fix errors before approving {task_id}.")
        return False

    # 2. Clean scratch files
    cleanup_transient_files()

    state = load_state()
    state["tasks"][task_id]["status"] = "completed"

    task = TASKS[task_id]
    phase_num = task["phase"]

    # 3. Advance to next task
    task_keys = list(TASKS.keys())
    curr_idx = task_keys.index(task_id)
    if curr_idx + 1 < len(task_keys):
        next_task = task_keys[curr_idx + 1]
        state["active_task"] = next_task
        state["current_phase"] = TASKS[next_task]["phase"]
    else:
        state["active_task"] = None
        print("\n🎉 ALL PHASES COMPLETE! Version 1.0 ready for release!")

    save_state(state)

    # 4. Conventional Git commit
    commit_msg = f"feat(phase{phase_num}): complete {task_id} - {task['title']}"
    subprocess.run(["git", "add", "."], cwd=str(WORKSPACE))
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(WORKSPACE))
    print(f"\n[Project Manager] Task {task_id} committed ({commit_msg}).")

    # 5. Remote synchronization
    if auto_sync:
        sync_git(push=True)

    print(f"[Project Manager] Active milestone advanced to: {state.get('active_task')}.")
    return True


def print_status():
    state = load_state()
    print("=" * 60)
    print("   RAG KNOWLEDGE GRAPH BUILDER - PROJECT MANAGER STATUS")
    print("=" * 60)
    print(f"Current Phase: Phase {state['current_phase']}")
    print(f"Active Task:   {state['active_task'] or 'None (All Completed)'}")
    print("-" * 60)
    for tid, tinfo in TASKS.items():
        st = state["tasks"].get(tid, {}).get("status", "pending")
        icon = "[x]" if st == "completed" else ("[~]" if st in ("in_progress", "review_passed") else "[ ]")
        active_mark = " <-- CURRENT" if tid == state["active_task"] else ""
        print(f" {icon} Phase {tinfo['phase']} | {tid}: {tinfo['title']:<38} [{st.upper()}]{active_mark}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Project Manager Pairing Agent for Ollama Coder")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show project status and current phase tasks")
    subparsers.add_parser("list", help="List all planned tasks")

    dispatch_p = subparsers.add_parser("dispatch", help="Dispatch task to Ollama coder agent")
    dispatch_p.add_argument("task_id", nargs="?", help="Task ID (default: active task)")
    dispatch_p.add_argument("--feedback", help="Reviewer feedback or revision instructions to append to prompt")

    prompt_p = subparsers.add_parser("prompt", help="Display work packet prompt for a task")
    prompt_p.add_argument("task_id", nargs="?", help="Task ID (default: active task)")

    review_p = subparsers.add_parser("review", help="Run local Ollama code review on current task")
    review_p.add_argument("task_id", nargs="?", help="Task ID (default: active task)")

    subparsers.add_parser("test", help="Run local test suite")
    subparsers.add_parser("sync", help="Synchronize with remote git repository")

    approve_p = subparsers.add_parser("approve", help="Approve task, commit code, and advance roadmap")
    approve_p.add_argument("task_id", nargs="?", help="Task ID (default: active task)")

    args = parser.parse_args()

    state = load_state()
    active_tid = state.get("active_task", "task_1_1")

    if args.command == "status" or not args.command:
        print_status()
    elif args.command == "list":
        for tid, tinfo in TASKS.items():
            print(f"{tid} (Phase {tinfo['phase']}): {tinfo['title']}")
    elif args.command == "sync":
        sync_git(push=True)
    elif args.command == "prompt":
        tid = args.task_id or active_tid
        t = TASKS[tid]
        print(f"\n--- WORK PACKET: {tid} ---")
        print(f"Title: {t['title']}")
        print(f"Goal: {t['description']}")
        print("Target Files:\n" + "\n".join(f"  - {f}" for f in t["target_files"]))
        print("Acceptance Criteria:\n" + "\n".join(f"  - {c}" for c in t["acceptance_criteria"]))
    elif args.command == "dispatch":
        tid = args.task_id or active_tid
        run_opencode_task(tid, feedback=args.feedback)
    elif args.command == "review":

        tid = args.task_id or active_tid
        run_local_review(tid)
    elif args.command == "test":
        run_local_tests()
    elif args.command == "approve":
        tid = args.task_id or active_tid
        approve_task(tid)


if __name__ == "__main__":
    main()
