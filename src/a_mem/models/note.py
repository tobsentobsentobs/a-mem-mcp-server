"""
Data models for Memory Notes
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class AtomicNote(BaseModel):
    """The core data model for a stored memory."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    contextual_summary: str = ""
    keywords: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    type: Optional[str] = Field(default=None, description="Node type: rule, procedure, concept, tool, reference, integration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Experimental fields for schema evolution")

class NoteInput(BaseModel):
    """Input vom User via MCP Tool."""
    content: str
    source: Optional[str] = "user_input"

class NoteRelation(BaseModel):
    """Definiert eine Kante im Graphen."""
    source_id: str
    target_id: str
    relation_type: str = Field(..., description="z.B. relates_to, contradicts, supports")
    reasoning: Optional[str] = "No reasoning provided"
    weight: float = 1.0
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when relation was created")

class SearchResult(BaseModel):
    note: AtomicNote
    score: float
    related_notes: List[AtomicNote] = []



