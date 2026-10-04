from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from rag_builder.api.routes import router
from rag_builder.api.state import app_state
from rag_builder.storage.memory import MemoryGraphStorage
import os

app = FastAPI(
    title="RAG Knowledge Graph Builder API",
    description="API for document ingestion and knowledge graph management",
    version="0.1.0"
)

# Initialize app_state with default MemoryGraphStorage
app_state.set_storage(MemoryGraphStorage())

app.include_router(router)

# Serve static files for the web UI
web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(web_dir):
    app.mount("/web", StaticFiles(directory=web_dir), name="web")

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Builder API. Use /docs for API documentation. Visit /web/index.html for visualization."}
