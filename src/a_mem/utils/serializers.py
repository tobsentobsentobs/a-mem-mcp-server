"""
Serialization helpers for exposing memory data via CLI/MCP tools.
"""

from __future__ import annotations

from typing import Any, Dict

from ..models.note import AtomicNote


def serialize_note(note: AtomicNote) -> Dict[str, Any]:
    """
    Returns a JSON-serializable dict representation of an AtomicNote.

    Dates are converted to ISO strings so that webviews and CLIs
    don't have to handle datetime objects manually.
    """
    data = note.model_dump()
    data["created_at"] = note.created_at.isoformat()
    return data

