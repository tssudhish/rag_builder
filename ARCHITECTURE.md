# Architecture: RAG Knowledge Graph Builder

## Overview
The application follows a pipeline architecture that transforms unstructured text into a structured graph representation (Nodes and Edges) to improve RAG retrieval accuracy by preserving semantic relationships.

## System Components

### 1. Ingestion Layer
- **Document Loaders**: Support for PDF, Markdown, TXT, and HTML.
- **Text Splitter**: Recursive character splitting to maintain context windows for LLM processing.

### 2. Extraction Layer (The Agent)
- **Entity Extractor**: Identifies key concepts, people, organizations, and objects using an LLM.
- **Relationship Extractor**: Identifies predicates (e.g., "is_a", "works_for", "located_in") connecting entities.
- **Schema Alignment**: Normalizes entity types and relation labels to prevent graph duplication.

### 3. Graph Storage Layer
- **Graph Database**: Use of Neo4j or FalkorDB for native graph queries.
- **Vector Indexing**: Hybrid approach where nodes store embedding vectors for semantic search alongside structural links.

### 4. RAG Integration Layer
- **Cypher/Gremlin Query Generator**: Converts natural language questions into graph queries.
- **Sub-graph Retrieval**: Fetches the target node and its $k$-hop neighbors to provide rich context to the LLM.

## Data Flow
`Document` $\rightarrow$ `Chunks` $\rightarrow$ `(Entity, Relation, Entity) Triplets` $\rightarrow$ `Graph DB` $\rightarrow$ `Contextual Sub-graph` $\rightarrow$ `LLM`
