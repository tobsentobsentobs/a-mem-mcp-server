"""
A-MEM CLI helper for external integrations.

Provides read/write access to the Graph + Vector stores so tools can inspect
and update memories without going through the MCP server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from a_mem.models.note import AtomicNote, NoteRelation
from a_mem.storage.engine import StorageManager
from a_mem.utils.serializers import serialize_note


def list_notes(manager: StorageManager) -> Dict[str, Any]:
    notes: List[Dict[str, Any]] = []
    for node_id, attrs in manager.graph.graph.nodes(data=True):
        try:
            note = AtomicNote(**attrs)
            notes.append(serialize_note(note))
        except Exception as exc:  # pragma: no cover - defensive
            notes.append({"id": node_id, "error": str(exc)})
    return {"notes": notes}


def get_graph(manager: StorageManager) -> Dict[str, Any]:
    nodes = []
    for node_id, attrs in manager.graph.graph.nodes(data=True):
        node = {"id": node_id}
        node.update({k: v for k, v in attrs.items() if k != "id"})
        nodes.append(node)

    edges = []
    for source, target, attrs in manager.graph.graph.edges(data=True):
        edges.append(
            {
                "source": source,
                "target": target,
                "relation_type": attrs.get("type", "relates_to"),
                "reasoning": attrs.get("reasoning", ""),
                "weight": attrs.get("weight", 1.0),
            }
        )
    return {"nodes": nodes, "edges": edges}


def get_note(manager: StorageManager, note_id: str) -> Dict[str, Any]:
    note = manager.get_note(note_id)
    if not note:
        return {"error": f"Note '{note_id}' not found"}
    return {"note": serialize_note(note)}


def list_relations(manager: StorageManager, note_id: str | None = None) -> Dict[str, Any]:
    edges: List[Dict[str, Any]] = []
    for source, target, attrs in manager.graph.graph.edges(data=True):
        if note_id and source != note_id and target != note_id:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "relation_type": attrs.get("type", "relates_to"),
                "reasoning": attrs.get("reasoning", ""),
                "weight": attrs.get("weight", 1.0),
            }
        )
    return {"relations": edges}


def remove_relation(
    manager: StorageManager,
    source: str,
    target: str,
) -> Dict[str, Any]:
    if not manager.graph.graph.has_edge(source, target):
        return {"error": f"Relation {source}->{target} not found"}
    manager.graph.graph.remove_edge(source, target)
    manager.graph.save_snapshot()
    return {"status": "edge_removed", "source": source, "target": target}


def update_note(manager: StorageManager, note_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    note = manager.get_note(note_id)
    if not note:
        return {"error": f"Note '{note_id}' not found"}

    if "content" in payload:
        return {
            "error": "Content edits require re-embedding and are not supported via CLI.",
        }

    allowed_fields = {"contextual_summary", "tags", "keywords"}
    updates = {k: v for k, v in payload.items() if k in allowed_fields}
    if not updates:
        return {"error": "No valid fields provided"}

    updated = note.model_copy(update=updates)
    manager.graph.update_node(updated)
    manager.graph.save_snapshot()

    return {"note": serialize_note(updated)}


def delete_note(manager: StorageManager, note_id: str) -> Dict[str, Any]:
    removed = manager.delete_note(note_id)
    if removed:
        manager.graph.save_snapshot()
        return {"status": "deleted", "note_id": note_id}
    return {"error": f"Note '{note_id}' not found"}


def add_relation(
    manager: StorageManager,
    source: str,
    target: str,
    relation_type: str,
    reasoning: str,
    weight: float,
) -> Dict[str, Any]:
    if source not in manager.graph.graph.nodes:
        return {"error": f"Source '{source}' not found"}
    if target not in manager.graph.graph.nodes:
        return {"error": f"Target '{target}' not found"}

    relation = NoteRelation(
        source_id=source,
        target_id=target,
        relation_type=relation_type,
        reasoning=reasoning,
        weight=weight,
    )
    manager.graph.add_edge(relation)
    manager.graph.save_snapshot()
    return {"status": "edge_added", "relation": relation.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description="A-MEM CLI helper")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list-notes")
    sub.add_parser("get-graph")
    note_cmd = sub.add_parser("get-note")
    note_cmd.add_argument("--id", required=True)

    relations_cmd = sub.add_parser("list-relations")
    relations_cmd.add_argument("--id", required=False, help="Optional note id filter")

    update = sub.add_parser("update-note")
    update.add_argument("--id", required=True, help="Note ID")
    update.add_argument("--data", required=True, help="JSON payload with fields to update")

    delete = sub.add_parser("delete-note")
    delete.add_argument("--id", required=True)

    relation = sub.add_parser("add-relation")
    relation.add_argument("--source", required=True)
    relation.add_argument("--target", required=True)
    relation.add_argument("--type", default="relates_to")
    relation.add_argument("--reasoning", default="Manual link")
    relation.add_argument("--weight", type=float, default=1.0)

    remove = sub.add_parser("remove-relation")
    remove.add_argument("--source", required=True)
    remove.add_argument("--target", required=True)

    args = parser.parse_args()
    manager = StorageManager()

    if args.command == "list-notes":
        result = list_notes(manager)
    elif args.command == "get-graph":
        result = get_graph(manager)
    elif args.command == "get-note":
        result = get_note(manager, args.id)
    elif args.command == "list-relations":
        result = list_relations(manager, getattr(args, "id", None))
    elif args.command == "update-note":
        try:
            payload = json.loads(args.data)
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"Invalid JSON payload: {exc}"}))
            sys.exit(1)
        result = update_note(manager, args.id, payload)
    elif args.command == "delete-note":
        result = delete_note(manager, args.id)
    elif args.command == "add-relation":
        result = add_relation(
            manager,
            args.source,
            args.target,
            args.type,
            args.reasoning,
            args.weight,
        )
    elif args.command == "remove-relation":
        result = remove_relation(manager, args.source, args.target)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

