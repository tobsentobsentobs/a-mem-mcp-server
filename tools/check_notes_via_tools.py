"""Check notes using A-MEM MCP tools directly."""

import sys
import asyncio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from a_mem.main import call_tool

async def check_notes():
    """Check notes using MCP tools."""
    print("=" * 60)
    print("📝 Checking Notes via A-MEM Tools")
    print("=" * 60)
    print()
    
    # Step 1: List all notes
    print("1️⃣ Listing all notes...")
    list_result = await call_tool("list_notes", {})
    
    if list_result and len(list_result) > 0:
        content = list_result[0].text if hasattr(list_result[0], 'text') else str(list_result[0])
        data = json.loads(content)
        
        notes = data.get("notes", [])
        print(f"   Found {len(notes)} notes")
        print()
        
        # Filter for async/asyncio notes
        async_notes = [
            n for n in notes
            if 'async' in str(n.get('contextual_summary', '')).lower()
            or 'asyncio' in str(n.get('content', '')).lower()
        ]
        
        if async_notes:
            print(f"   Found {len(async_notes)} async-related notes")
            print()
            
            # Show first 2 notes in detail
            for i, note in enumerate(async_notes[:2], 1):
                note_id = note.get('id', '')
                summary = note.get('contextual_summary', '')
                
                print(f"2️⃣ Getting full details for note {i}...")
                print(f"   ID: {note_id[:8]}...")
                print()
                
                # Get full note details
                get_result = await call_tool("get_note", {"note_id": note_id})
                
                if get_result and len(get_result) > 0:
                    note_content = get_result[0].text if hasattr(get_result[0], 'text') else str(get_result[0])
                    note_data = json.loads(note_content)
                    
                    note_info = note_data.get("note", {})
                    full_summary = note_info.get("contextual_summary", "")
                    
                    print(f"   ✅ Full Summary ({len(full_summary)} chars):")
                    print(f"   {full_summary}")
                    print()
                    print(f"   Type: {note_info.get('type', 'N/A')}")
                    print(f"   Keywords: {', '.join(note_info.get('keywords', [])[:5])}")
                    print(f"   Tags: {', '.join(note_info.get('tags', [])[:5])}")
                    print()
                    print("=" * 60)
                    print()
        else:
            print("   No async-related notes found")
            print("   Showing first 2 notes instead:")
            print()
            
            for i, note in enumerate(notes[:2], 1):
                note_id = note.get('id', '')
                summary = note.get('contextual_summary', '')
                
                print(f"2️⃣ Getting full details for note {i}...")
                print(f"   ID: {note_id[:8]}...")
                print()
                
                # Get full note details
                get_result = await call_tool("get_note", {"note_id": note_id})
                
                if get_result and len(get_result) > 0:
                    note_content = get_result[0].text if hasattr(get_result[0], 'text') else str(get_result[0])
                    note_data = json.loads(note_content)
                    
                    note_info = note_data.get("note", {})
                    full_summary = note_info.get("contextual_summary", "")
                    
                    print(f"   ✅ Full Summary ({len(full_summary)} chars):")
                    print(f"   {full_summary}")
                    print()
                    print(f"   Type: {note_info.get('type', 'N/A')}")
                    print(f"   Keywords: {', '.join(note_info.get('keywords', [])[:5])}")
                    print(f"   Tags: {', '.join(note_info.get('tags', [])[:5])}")
                    print()
                    print("=" * 60)
                    print()

if __name__ == "__main__":
    asyncio.run(check_notes())

