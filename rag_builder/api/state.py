from typing import Dict, Any, Optional
from threading import Lock
from dataclasses import dataclass, field
import uuid

@dataclass
class DocumentState:
    document_id: str
    filename: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float = 0.0
    error: Optional[str] = None

class AppState:
    """
    Thread-safe state manager for tracking document processing and graph metrics.
    """
    def __init__(self):
        self._lock = Lock()
        self._documents: Dict[str, DocumentState] = {}
        self._graph_stats: Dict[str, Any] = {
            "node_count": 0,
            "edge_count": 0,
            "last_updated": None
        }

    def update_document_status(self, doc_id: str, filename: str = None, status: str = None, progress: float = None, error: str = None):
        with self._lock:
            if doc_id not in self._documents:
                # If filename is not provided but it's a new doc, we can't create it properly
                # but usually, we initialize it first.
                if filename is None:
                    return
                self._documents[doc_id] = DocumentState(document_id=doc_id, filename=filename, status='pending')

            doc = self._documents[doc_id]
            if status is not None:
                doc.status = status
            if progress is not None:
                doc.progress = progress
            if error is not None:
                doc.error = error

    def get_document_status(self, doc_id: str) -> Optional[DocumentState]:
        with self._lock:
            return self._documents.get(doc_id)

    def update_graph_stats(self, node_count: int = None, edge_count: int = None):
        with self._lock:
            if node_count is not None:
                self._graph_stats["node_count"] = node_count
            if edge_count is not None:
                self._graph_stats["edge_count"] = edge_count
            
            import datetime
            self._graph_stats["last_updated"] = datetime.datetime.utcnow().isoformat()

    def get_graph_stats(self) -> Dict[str, Any]:
        with self._lock:
            return self._graph_stats.copy()

# Global state instance
app_state = AppState()
