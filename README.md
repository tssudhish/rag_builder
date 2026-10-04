# RAG Knowledge Graph Builder v1.0

RAG Knowledge Graph Builder is a powerful pipeline that converts unstructured documents into a structured knowledge graph to enhance Retrieval-Augmented Generation (RAG) pipelines. By preserving semantic relationships between entities, it enables more accurate and context-aware retrieval compared to standard vector-based RAG.

## 🚀 Key Features

- **Multi-format Ingestion**: Support for PDF, Markdown, TXT, and HTML.
- **LLM-Powered Extraction**: Uses specialized models (like `sciphi-triplex`) to extract high-quality (Subject, Predicate, Object) triplets.
- **Graph Persistence**: Save and load knowledge graphs to/from JSON files.
- **Hybrid Retrieval**: Combines graph traversal (k-hop neighborhood) with LLM generation for grounded answers.
- **CLI Interface**: Simple command-line tool for building and querying graphs.

## 🛠 Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally.
- Required models pulled:
  ```bash
  ollama pull sciphi-triplex
  ollama pull gemma2
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/rag_builder.git
   cd rag_builder
   ```
2. Install dependencies:
   ```bash
   pip install -e .
   ```

## 💻 CLI Usage

The `rag-builder` command provides the primary interface.

### 1. Build a Knowledge Graph
Extract triplets from a document and optionally save the graph to disk.
```bash
rag-builder build --file path/to/document.pdf --model sciphi-triplex --save my_graph.json
```
- `--file`: Path to the source document.
- `--model`: Model used for extraction (default: `sciphi-triplex`).
- `--save`: (Optional) Path to save the graph as a JSON file.

### 2. Query the Knowledge Graph
Ask a question based on a previously built graph or a document.
```bash
# Using a saved graph
rag-builder query --query "What is the relationship between X and Y?" --load my_graph.json

# Building on the fly from a file
rag-builder query --query "What is the relationship between X and Y?" --file path/to/document.pdf
```
- `--query`: The natural language question.
- `--load`: Path to a saved graph JSON.
- `--file`: Path to a document to build a temporary graph from.
- `--model`: Model used for generation (default: `gemma2`).

## 🏗 Architecture

The system follows a linear pipeline:
`Document` $\rightarrow$ `Chunks` $\rightarrow$ `(Entity, Relation, Entity) Triplets` $\rightarrow$ `Graph DB` $\rightarrow$ `Contextual Sub-graph` $\rightarrow$ `LLM`

1. **Ingestion**: Documents are loaded and split into manageable chunks.
2. **Extraction**: An LLM extracts semantic triplets from each chunk.
3. **Storage**: Triplets are inserted into a graph (MemoryGraphStorage by default).
4. **Retrieval**: Natural language queries are processed to identify target entities, and their k-hop neighborhood is retrieved.
5. **Generation**: The retrieved sub-graph is provided as context to an LLM to generate a grounded response with citations.

## 📈 Roadmap
- [ ] Integration with Neo4j/FalkorDB for production-scale graphs.
- [ ] Advanced Cypher query generation for complex multi-hop questions.
- [ ] Web UI for graph visualization and management.
