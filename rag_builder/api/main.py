from fastapi import FastAPI
from rag_builder.api.routes import router

app = FastAPI(
    title="RAG Knowledge Graph Builder API",
    description="API for document ingestion and knowledge graph management",
    version="0.1.0"
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Builder API. Use /docs for API documentation."}
