# Project Manager Pairing Agent Architecture: `rag_builder`

## 1. Overview & Strategy

This system pairs a high-level **Project Manager (PM)** supervisory agent with local **Ollama** agents to build `rag_builder` through all planned phases up to Version 1.0.

### Token Conservation Policy
- **Zero Gemini Code-Writing Overhead**: All code writing, refactoring, boilerplate generation, and debugging is offloaded to the local **Coder** agent powered by Ollama.
- **Local Review & Test Loops**: Code reviews and test setups are executed locally using Ollama (`gemma4:31b-cloud` / local subagents), preventing costly Gemini token roundtrips during iterative development loops.
- **PM High-Level Oversight**: The Gemini Project Manager acts strictly as an architectural supervisor: validating work packets, proofreading git diffs, monitoring acceptance criteria, and managing phase transitions.

---

## 2. Agent Roles & Topology

```
                  ┌────────────────────────────────────────┐
                  │       Gemini AI / Antigravity          │
                  │        "project_manager"               │
                  │  (Supervision, Proofreading, Roadmap)  │
                  └──────────────────┬─────────────────────┘
                                     │ Issues Work Packets & Commands
                                     ▼
                  ┌────────────────────────────────────────┐
                  │       Local PM Orchestrator            │
                  │       (pm_orchestrator.py)             │
                  └──────┬───────────┬──────────────┬──────┘
                         │           │              │
       Dispatches Tasks  │           │ Reviews Code │ Runs Test Suite
                         ▼           ▼              ▼
                 ┌──────────────┐  ┌─────────────┐  ┌────────────┐
                 │    coder     │  │  reviewer   │  │   tester   │
                 │ (OpenCode +  │  │   (Ollama   │  │  (Ollama   │
                 │   Ollama)    │  │  Gemma 4)   │  │  Gemma 4)  │
                 └──────┬───────┘  └─────────────┘  └────────────┘
                        │
                        ▼ (Extraction Validation)
                 ┌──────────────────┐
                 │ triplet_extractor│
                 │ (SciPhi Triplex) │
                 └──────────────────┘
```

| Agent Name | Engine / Model | Role & Permissions |
| :--- | :--- | :--- |
| **`project_manager`** | Gemini 3.7 / Antigravity | Orchestrates phases, specifies technical requirements, monitors git progress, and ensures adherence to [ARCHITECTURE.md](file:///c:/Users/<username>/code/rag_builder/ARCHITECTURE.md). |
| **`coder`** | OpenCode (`ollama/gemma4:31b-cloud`) | Primary implementation agent. Generates files, edits code, installs packages, runs scripts. |
| **`reviewer`** | Ollama (`gemma4:31b-cloud`) | Autonomous code reviewer. Analyzes git diffs for bugs, schema mismatches, type hints, and code smell. |
| **`tester`** | Ollama (`gemma4:31b-cloud`) | Generates test cases, configures test runners (e.g. `pytest`/`unittest`), and verifies test execution. |
| **`triplet_extractor`** | Ollama (`sciphi/triplex:latest`) | Specialist model used for validating triplet extraction from text chunks during Phase 1. |

---

## 3. Configuration Details

- **Ollama Location**: `C:\Users\<username>\AppData\Local\Programs\Ollama\ollama.exe`
- **Models Root**: `C:\Users\<username>\.ollama\models`
- **Active Ollama Server**: `http://localhost:11434`
- **OpenCode Config**: [opencode.json](file:///c:/Users/<username>/code/rag_builder/opencode.json) in workspace root.
- **Python Environment**: `D:\Users\<username>\miniconda3\python.exe`

---

## 4. Operational Workflow (DevOps Loop)

For every task defined in [ROADMAP.md](file:///c:/Users/<username>/code/rag_builder/ROADMAP.md):

1. **Prompt Generation**:
   The Project Manager generates a formal **Work Packet** specifying:
   - Target files to create/modify
   - Architecture context
   - Acceptance criteria and typing constraints
   ```powershell
   .\pm.ps1 prompt <task_id>
   ```

2. **Dispatch to Coder**:
   The PM dispatcher invokes the `coder` agent via OpenCode:
   ```powershell
   .\pm.ps1 dispatch <task_id>
   ```

3. **Autonomous Local Review**:
   The local `reviewer` evaluates git diffs and code structure:
   ```powershell
   .\pm.ps1 review <task_id>
   ```

4. **Autonomous Local Testing**:
   The local `tester` runs unit and integration tests:
   ```powershell
   .\pm.ps1 test
   ```

5. **Approval & Roadmap Advancement**:
   Once verified, PM marks the task completed, commits changes to Git, and advances the active pointer to the next roadmap milestone:
   ```powershell
   .\pm.ps1 approve <task_id>
   ```

---

## 5. Phase-to-v1.0 Milestone Checklist

- [ ] **Phase 1: Core Extraction Engine**
  - [ ] `task_1_1`: Document Loaders (PDF, MD, TXT, HTML) & Text Splitter
  - [ ] `task_1_2`: LLM Triplet Extraction Engine (`sciphi/triplex`)
  - [ ] `task_1_3`: Schema Alignment & Normalization
  - [ ] `task_1_4`: Phase 1 Validation & Test Harness
- [ ] **Phase 2: Graph Storage & Schema**
  - [ ] `task_2_1`: Graph Database Interface & Schema
  - [ ] `task_2_2`: Triplet Ingestion Pipeline & Vector Indexing
  - [ ] `task_2_3`: Graph Verification & Inspection Queries
- [ ] **Phase 3: Backend API & Web Integration**
  - [ ] `task_3_1`: FastAPI Backend & Document Upload Endpoints
  - [ ] `task_3_2`: Graph Visualization API & Web UI (Cytoscape/D3)
- [ ] **Phase 4: GraphRAG Implementation**
  - [ ] `task_4_1`: Natural Language to Graph Query Translation
  - [ ] `task_4_2`: Sub-graph Retrieval ($k$-hop neighborhood)
  - [ ] `task_4_3`: Context-Aware Response Generation & Evaluation
- [ ] **Phase 5: Agent Export & DevOps -> v1.0**
  - [ ] `task_5_1`: Agent Tool Definitions (LangChain / Function Calling)
  - [ ] `task_5_2`: End-to-End DevOps, Test Coverage & Packaging (v1.0 Release)
