# Project Roadmap: RAG Knowledge Graph Builder

## Phase 1: Core Extraction Engine [ ]
- [ ] Implement document loaders (PDF, TXT, MD)
- [ ] Develop LLM-based entity and relationship extraction logic
- [ ] Implement triplet generation (Subject -> Predicate -> Object)
- [ ] Validation of extraction quality

## Phase 2: Graph Storage & Schema [ ]
- [ ] Select and setup Graph Database (e.g., Neo4j)
- [ ] Define Graph Schema (Node labels, Relationship types)
- [ ] Build the ingestion pipeline from triplets to Graph DB
- [ ] Implement basic graph verification queries

## Phase 3: Backend API & Web Integration [ ]
- [ ] Set up Backend API (FastAPI/Flask)
- [ ] Create document upload and processing endpoints
- [ ] Integrate a basic frontend for progress tracking
- [ ] Integrate a graph visualization library (e.g., Cytoscape.js or D3.js)

## Phase 4: GraphRAG Implementation [ ]
- [ ] Implement NL to Cypher/Gremlin query translation
- [ ] Develop sub-graph retrieval logic (k-hop neighborhood)
- [ ] Integrate with LLM for context-aware response generation
- [ ] Evaluate retrieval accuracy vs standard RAG

## Phase 5: Agent Export & Integration [ ]
- [ ] Create API endpoints for external AI Agents to query the KG
- [ ] Implement GraphRAG tool definitions for Agent frameworks (e.g., LangChain/AutoGPT)
- [ ] Final end-to-end testing and optimization
