"""
Test Suite für neue Features: Type Classification, Priority Scoring, Event Logging

Tests für:
1. Type Classification in extract_metadata
2. Priority Scoring
3. Event Logging
4. Metadata Field Support
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from a_mem.models.note import AtomicNote
from a_mem.utils.priority import compute_priority, log_event, keyword_prefilter


class TestTypeClassification:
    """Test für Type-Klassifikation"""
    
    def test_atomic_note_with_type(self):
        """Test: AtomicNote kann type-Feld haben"""
        note = AtomicNote(
            content="Test content",
            type="rule"
        )
        
        assert note.type == "rule"
        assert note.metadata == {}  # Default empty dict
        print("✅ AtomicNote mit type-Feld funktioniert")
    
    def test_atomic_note_with_metadata(self):
        """Test: AtomicNote kann metadata-Feld haben"""
        note = AtomicNote(
            content="Test content",
            metadata={"experimental_field": "value", "custom_data": 123}
        )
        
        assert note.metadata["experimental_field"] == "value"
        assert note.metadata["custom_data"] == 123
        print("✅ AtomicNote mit metadata-Feld funktioniert")
    
    def test_atomic_note_backward_compatibility(self):
        """Test: Alte Notes ohne type/metadata funktionieren noch"""
        note = AtomicNote(
            content="Test content",
            contextual_summary="Summary",
            keywords=["k1"],
            tags=["t1"]
        )
        
        # Beide Felder sollten None/empty sein, aber Note sollte funktionieren
        assert note.type is None or note.type == "concept"  # Default könnte "concept" sein
        assert note.metadata == {}
        print("✅ Backward Compatibility: Alte Notes funktionieren")


class TestPriorityScoring:
    """Test für Priority-Scoring"""
    
    def test_priority_rule_type(self):
        """Test: Rules haben höhere Priority"""
        rule_note = AtomicNote(
            content="Never use eval() in production",
            type="rule",
            created_at=datetime.now()
        )
        
        concept_note = AtomicNote(
            content="Python is a programming language",
            type="concept",
            created_at=datetime.now()
        )
        
        rule_priority = compute_priority(rule_note, usage_count=0, edge_count=0)
        concept_priority = compute_priority(concept_note, usage_count=0, edge_count=0)
        
        assert rule_priority > concept_priority
        print(f"✅ Rule Priority ({rule_priority}) > Concept Priority ({concept_priority})")
    
    def test_priority_with_usage(self):
        """Test: Usage erhöht Priority"""
        note = AtomicNote(
            content="Test",
            type="concept",
            created_at=datetime.now()
        )
        
        priority_no_usage = compute_priority(note, usage_count=0, edge_count=0)
        priority_with_usage = compute_priority(note, usage_count=5, edge_count=0)
        
        assert priority_with_usage > priority_no_usage
        print(f"✅ Priority mit Usage ({priority_with_usage}) > ohne Usage ({priority_no_usage})")
    
    def test_priority_with_edges(self):
        """Test: Edge Count erhöht Priority"""
        note = AtomicNote(
            content="Test",
            type="concept",
            created_at=datetime.now()
        )
        
        priority_no_edges = compute_priority(note, usage_count=0, edge_count=0)
        priority_with_edges = compute_priority(note, usage_count=0, edge_count=5)
        
        assert priority_with_edges > priority_no_edges
        print(f"✅ Priority mit Edges ({priority_with_edges}) > ohne Edges ({priority_no_edges})")
    
    def test_priority_age_factor(self):
        """Test: Ältere Notes haben niedrigere Priority"""
        old_note = AtomicNote(
            content="Test",
            type="concept",
            created_at=datetime.now() - timedelta(days=100)
        )
        
        new_note = AtomicNote(
            content="Test",
            type="concept",
            created_at=datetime.now()
        )
        
        old_priority = compute_priority(old_note, usage_count=0, edge_count=0)
        new_priority = compute_priority(new_note, usage_count=0, edge_count=0)
        
        # Alte Note sollte niedrigere Priority haben (aber nicht zu niedrig wegen min 0.3)
        assert old_priority < new_priority
        assert old_priority >= 0.3  # Minimum age factor
        print(f"✅ Neue Note Priority ({new_priority}) > Alte Note Priority ({old_priority})")


class TestEventLogging:
    """Test für Event-Logging"""
    
    def test_event_logging_creates_file(self):
        """Test: Event-Logging erstellt Datei"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporärer Event-Log-Pfad
            from a_mem.utils import priority
            original_path = priority.EVENT_LOG_PATH
            priority.EVENT_LOG_PATH = Path(tmpdir) / "test_events.jsonl"
            
            try:
                # Log Event
                log_event("TEST_EVENT", {"test": "data", "value": 123})
                
                # Prüfe ob Datei existiert
                assert priority.EVENT_LOG_PATH.exists()
                
                # Prüfe Inhalt
                with open(priority.EVENT_LOG_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    assert len(lines) == 1
                    
                    entry = json.loads(lines[0])
                    assert entry["event"] == "TEST_EVENT"
                    assert entry["data"]["test"] == "data"
                    assert entry["data"]["value"] == 123
                    assert "timestamp" in entry
                
                print("✅ Event-Logging erstellt korrekte JSONL-Datei")
            finally:
                priority.EVENT_LOG_PATH = original_path
    
    def test_event_logging_append_only(self):
        """Test: Event-Logging ist append-only"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from a_mem.utils import priority
            original_path = priority.EVENT_LOG_PATH
            priority.EVENT_LOG_PATH = Path(tmpdir) / "test_events.jsonl"
            
            try:
                # Mehrere Events loggen
                log_event("EVENT_1", {"data": 1})
                log_event("EVENT_2", {"data": 2})
                log_event("EVENT_3", {"data": 3})
                
                # Prüfe alle Events sind da
                with open(priority.EVENT_LOG_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    assert len(lines) == 3
                    
                    for i, line in enumerate(lines, 1):
                        entry = json.loads(line)
                        assert entry["event"] == f"EVENT_{i}"
                        assert entry["data"]["data"] == i
                
                print("✅ Event-Logging ist append-only (alle Events erhalten)")
            finally:
                priority.EVENT_LOG_PATH = original_path


class TestKeywordPrefilter:
    """Test für Keyword-Vorfilter"""
    
    def test_keyword_prefilter_matches(self):
        """Test: Keyword-Vorfilter findet Matches"""
        notes = [
            AtomicNote(content="Python async programming", keywords=["python", "async"]),
            AtomicNote(content="JavaScript promises", keywords=["javascript", "promises"]),
            AtomicNote(content="Python threading", keywords=["python", "threading"]),
        ]
        
        filtered = keyword_prefilter("Python async", notes)
        
        # Sollte nur Python-relevante Notes finden
        assert len(filtered) >= 1
        assert any("Python" in n.content for n in filtered)
        print(f"✅ Keyword-Vorfilter findet {len(filtered)} relevante Notes")
    
    def test_keyword_prefilter_empty_query(self):
        """Test: Leere Query gibt alle Notes zurück"""
        notes = [
            AtomicNote(content="Note 1"),
            AtomicNote(content="Note 2"),
        ]
        
        filtered = keyword_prefilter("", notes)
        assert len(filtered) == len(notes)
        print("✅ Keyword-Vorfilter mit leerer Query gibt alle Notes zurück")


def run_tests():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("🧪 A-MEM New Features Test Suite")
    print("="*60 + "\n")
    
    test_classes = [
        TestTypeClassification,
        TestPriorityScoring,
        TestEventLogging,
        TestKeywordPrefilter
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        print("-" * 60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"  ❌ {method_name}: {e}")
            except Exception as e:
                print(f"  ⚠️  {method_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
    print("="*60 + "\n")
    
    if passed_tests == total_tests:
        print("✅ ALLE TESTS BESTANDEN!")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} Test(s) fehlgeschlagen")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

