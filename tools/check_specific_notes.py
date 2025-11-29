"""Check specific notes by ID from research test."""

import sys
import asyncio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from a_mem.main import call_tool

async def check_specific_notes():
    """Check specific notes from research test."""
    print("=" * 60)
    print("📝 Checking Specific Notes from Research Test")
    print("=" * 60)
    print()
    
    # Note IDs from the test output
    note_ids = [
        "287eb61a-5657-48c7-aef0-affaeb3cb4fd",  # From test output
        "d4fe4e51-f073-45cf-b5a3-79744b411915",  # From test output
    ]
    
    for i, note_id in enumerate(note_ids, 1):
        print(f"{i}️⃣ Getting note: {note_id[:8]}...")
        print()
        
        try:
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
                
                # Check if summary was truncated in storage
                if len(full_summary) <= 50:
                    print(f"   ⚠️  Summary is only {len(full_summary)} chars - might be truncated!")
                else:
                    print(f"   ✅ Summary is complete ({len(full_summary)} chars)")
                print()
                print("=" * 60)
                print()
            else:
                print(f"   ❌ Note not found or error occurred")
                print()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

if __name__ == "__main__":
    asyncio.run(check_specific_notes())

