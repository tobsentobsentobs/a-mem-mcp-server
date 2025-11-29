"""Check if note summaries are stored completely."""

import json
from pathlib import Path

def check_summaries():
    """Check note summaries from graph file."""
    graph_file = Path("data/graph/knowledge_graph.json")
    
    if not graph_file.exists():
        print("Graph file not found!")
        return
    
    with open(graph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find notes with "async" or "asyncio" in summary or content
    notes = [
        n for n in data.get('nodes', [])
        if 'async' in str(n.get('contextual_summary', '')).lower() 
        or 'asyncio' in str(n.get('content', '')).lower()
        or 'asyncio' in str(n.get('contextual_summary', '')).lower()
    ]
    
    print("=" * 60)
    print("📝 Checking Note Summaries (Full Length)")
    print("=" * 60)
    print()
    
    for i, note in enumerate(notes[:5], 1):
        note_id = note.get('id', '')
        summary = note.get('contextual_summary', '')
        note_type = note.get('type', 'N/A')
        
        print(f"{i}. ID: {note_id[:8]}...")
        print(f"   Type: {note_type}")
        print(f"   Summary Length: {len(summary)} chars")
        print(f"   Full Summary: {summary}")
        print()

if __name__ == "__main__":
    check_summaries()
