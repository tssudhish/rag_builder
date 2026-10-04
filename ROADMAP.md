# Project Roadmap: RAG Knowledge Graph Builder

## Phase 1: Core Extraction Engine [x]
- [x] Implement document loaders (PDF, TXT, MD, HTML) & Text Splitter
- [x] Develop LLM-based entity and relationship extraction logic
- [x] Implement triplet generation (Subject -> Predicate -> Object)
- [x] Validation of extraction quality

## Phase 2: Graph Storage & Schema [x]
- [x] Select and setup Graph Database (e.g., Neo4j / In-Memory NetworkX)
- [x] Define Graph Schema (Node labels, Relationship types)
- [x] Build the ingestion pipeline from triplets to Graph DB
- [x] Implement basic graph verification queries

## Phase 3: Backend API & Web Integration [x]
- [x] Set up Backend API (FastAPI)
- [x] Create document upload and processing endpoints
- [x] Integrate a basic frontend for progress tracking
- [x] Integrate a graph visualization library (Cytoscape.js)

## Phase 4: GraphRAG Implementation [x]
- [x] Implement NL to Cypher/Gremlin query translation
- [x] Develop sub-graph retrieval logic (k-hop neighborhood)
- [x] Integrate with LLM for context-aware response generation
- [x] Evaluate retrieval accuracy vs standard RAG

## Phase 5: Agent Export & Integration [x]
- [x] Create API endpoints for external AI Agents to query the KG
- [x] Implement GraphRAG tool definitions for Agent frameworks (LangChain / Function Calling)
- [x] Final end-to-end testing, CLI packaging, and v1.0 release
