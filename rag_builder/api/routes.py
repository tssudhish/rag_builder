import os
import uuid
import shutil
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from rag_builder.api.state import app_state
from rag_builder.api.visualizer import GraphVisualizer

router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".html"}

class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    progress: float
    error: Optional[str] = None

class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int
    last_updated: str = None

def process_document_task(doc_id: str, file_path: str):
    """
    Background task to simulate document processing.
    In a real implementation, this would call the Ingestion and Extraction layers.
    """
    try:
        app_state.update_document_status(doc_id, status="processing", progress=0.1)
        
        # Simulate processing steps
        import time
        time.sleep(2) # Simulating work
        app_state.update_document_status(doc_id, status="processing", progress=0.5)
        
        time.sleep(2) # Simulating work
        app_state.update_document_status(doc_id, status="processing", progress=0.9)
        
        time.sleep(1) # Simulating work
        app_state.update_document_status(doc_id, status="completed", progress=1.0)
        
        # Update graph stats to simulate adding data
        app_state.update_graph_stats(node_count=10, edge_count=25)
        
    except Exception as e:
        app_state.update_document_status(doc_id, status="failed", error=str(e))

@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Create unique doc id and file path
    doc_id = str(uuid.uuid4())
    upload_dir = "data/uploads"
    file_path = os.path.join(upload_dir, f"{doc_id}{ext}")

    # Persist file to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {str(e)}"
        )

    # Initialize state
    app_state.update_document_status(doc_id, filename=file.filename, status="pending")

    # Queue background task
    background_tasks.add_task(process_document_task, doc_id, file_path)

    return {"document_id": doc_id, "message": "Document uploaded and queued for processing"}

@router.get("/documents/status/{doc_id}", response_model=DocumentStatusResponse)
async def get_document_status(doc_id: str):
    doc = app_state.get_document_status(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return DocumentStatusResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        status=doc.status,
        progress=doc.progress,
        error=doc.error
    )

@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    stats = app_state.get_graph_stats()
    return GraphStatsResponse(**stats)

@router.get("/graph/nodes-and-edges")
async def get_graph_visualization(limit: int = 100):
    """
    Returns the graph in Cytoscape.js compatible format.
    """
    storage = app_state.get_storage()
    if not storage:
        # Return empty graph if no storage is initialized
        return {"nodes": [], "edges": []}
    
    return GraphVisualizer.to_cytoscape_format(storage, limit=limit)
