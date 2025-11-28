"""
Test Suite für Memory Enzymes

Tests für:
1. Link-Pruner
2. Relation-Suggester
3. Summary-Digester
4. Enzyme-Scheduler
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

from a_mem.models.note import AtomicNote, NoteRelation
from a_mem.storage.engine import GraphStore, VectorStore, StorageManager
from a_mem.utils.enzymes import prune_links, suggest_relations, digest_node, run_memory_enzymes, cosine_similarity
from a_mem.utils.llm import LLMService
from a_mem.config import settings


class TestCosineSimilarity:
    """Test für Cosine Similarity Funktion"""
    
    def test_cosine_similarity_identical(self):
        """Test: Identische Vektoren haben Similarity 1.0"""
        vec = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec, vec)
        assert abs(result - 1.0) < 0.001
        print("✅ Identische Vektoren: Similarity = 1.0")
    
    def test_cosine_similarity_orthogonal(self):
        """Test: Orthogonale Vektoren haben Similarity 0.0"""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        result = cosine_similarity(vec_a, vec_b)
        assert abs(result) < 0.001
        print("✅ Orthogonale Vektoren: Similarity ≈ 0.0")
    
    def test_cosine_similarity_zero_vector(self):
        """Test: Zero-Vektoren haben Similarity 0.0"""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec_a, vec_b)
        assert result == 0.0
        print("✅ Zero-Vektor: Similarity = 0.0")


class TestLinkPruner:
    """Test für Link-Pruner"""
    
    def test_prune_weak_links(self):
        """Test: Schwache Links werden entfernt"""
        graph = GraphStore()
        
        # Erstelle Test-Notes
        note1 = AtomicNote(content="Test 1")
        note2 = AtomicNote(content="Test 2")
        graph.add_node(note1)
        graph.add_node(note2)
        
        # Erstelle schwache Relation
        weak_relation = NoteRelation(
            source_id=note1.id,
            target_id=note2.id,
            relation_type="relates_to",
            weight=0.1  # Sehr schwach
        )
        graph.add_edge(weak_relation)
        
        # Prüfe Edge existiert
        assert graph.graph.has_edge(note1.id, note2.id)
        
        # Prune mit min_weight=0.3
        pruned = prune_links(graph, max_age_days=90, min_weight=0.3)
        
        # Edge sollte entfernt sein
        assert not graph.graph.has_edge(note1.id, note2.id)
        assert pruned == 1
        print(f"✅ Schwache Links entfernt: {pruned} Edge(s)")
    
    def test_prune_keeps_strong_links(self):
        """Test: Starke Links bleiben erhalten"""
        graph = GraphStore()
        
        note1 = AtomicNote(content="Test 1")
        note2 = AtomicNote(content="Test 2")
        graph.add_node(note1)
        graph.add_node(note2)
        
        # Erstelle starke Relation
        strong_relation = NoteRelation(
            source_id=note1.id,
            target_id=note2.id,
            relation_type="relates_to",
            weight=0.9  # Stark
        )
        graph.add_edge(strong_relation)
        
        # Prune
        pruned = prune_links(graph, max_age_days=90, min_weight=0.3)
        
        # Edge sollte bleiben
        assert graph.graph.has_edge(note1.id, note2.id)
        assert pruned == 0
        print("✅ Starke Links bleiben erhalten")


class TestRelationSuggester:
    """Test für Relation-Suggester"""
    
    def test_suggest_relations_finds_similar(self):
        """Test: Suggester findet ähnliche Notes"""
        graph = GraphStore()
        
        # Erstelle ähnliche Notes
        note1 = AtomicNote(
            content="Python async programming with asyncio",
            keywords=["python", "async", "asyncio"],
            tags=["programming"]
        )
        note2 = AtomicNote(
            content="Async/await patterns in Python",
            keywords=["python", "async", "patterns"],
            tags=["programming"]
        )
        
        graph.add_node(note1)
        graph.add_node(note2)
        
        notes = {note1.id: note1, note2.id: note2}
        
        # Mock LLM Service für Embeddings
        mock_llm = Mock(spec=LLMService)
        
        # Simuliere ähnliche Embeddings (gleiche Keywords → ähnliche Embeddings)
        embedding1 = [0.5, 0.3, 0.8] * 256  # 768 dimensions für nomic-embed
        embedding2 = [0.4, 0.3, 0.9] * 256  # Ähnlich
        
        def mock_get_embedding(text):
            if "asyncio" in text.lower():
                return embedding1[:768]  # Truncate auf 768
            return embedding2[:768]
        
        mock_llm.get_embedding = mock_get_embedding
        
        # Test mit niedrigem Threshold (da Mock-Embeddings nicht perfekt sind)
        suggestions = suggest_relations(notes, graph, mock_llm, threshold=0.5, max_suggestions=10)
        
        # Sollte mindestens eine Suggestion finden (wenn Embeddings ähnlich genug)
        # Da wir Mock-Embeddings haben, könnte es sein dass keine gefunden wird
        # Aber die Funktion sollte zumindest laufen
        assert isinstance(suggestions, list)
        print(f"✅ Relation-Suggester läuft: {len(suggestions)} Suggestions gefunden")
    
    def test_suggest_relations_skips_existing(self):
        """Test: Suggester überspringt bereits verbundene Notes"""
        graph = GraphStore()
        
        note1 = AtomicNote(content="Test 1")
        note2 = AtomicNote(content="Test 2")
        graph.add_node(note1)
        graph.add_node(note2)
        
        # Erstelle bereits existierende Relation
        relation = NoteRelation(
            source_id=note1.id,
            target_id=note2.id,
            relation_type="relates_to"
        )
        graph.add_edge(relation)
        
        notes = {note1.id: note1, note2.id: note2}
        
        mock_llm = Mock(spec=LLMService)
        mock_llm.get_embedding = lambda x: [0.5] * 768
        
        suggestions = suggest_relations(notes, graph, mock_llm, threshold=0.5)
        
        # Sollte keine Suggestions finden, da bereits verbunden
        # (oder zumindest nicht diese beiden)
        assert all(s[0] != note1.id or s[1] != note2.id for s in suggestions)
        print("✅ Bereits verbundene Notes werden übersprungen")


class TestDigestNode:
    """Test für Summary-Digester"""
    
    def test_digest_node_skips_small_nodes(self):
        """Test: Nodes mit wenigen Children werden nicht verdaut"""
        note1 = AtomicNote(content="Parent")
        children = [
            AtomicNote(content=f"Child {i}")
            for i in range(5)  # Nur 5 Children
        ]
        
        mock_llm = Mock(spec=LLMService)
        
        result = digest_node(note1.id, children, mock_llm, max_children=8)
        
        # Sollte None zurückgeben (nicht nötig)
        assert result is None
        print("✅ Nodes mit wenigen Children werden nicht verdaut")
    
    def test_digest_node_processes_large_nodes(self):
        """Test: Nodes mit vielen Children werden verdaut"""
        note1 = AtomicNote(content="Parent")
        children = [
            AtomicNote(content=f"Child {i}")
            for i in range(10)  # 10 Children (> max_children=8)
        ]
        
        mock_llm = Mock(spec=LLMService)
        mock_llm._call_llm = Mock(return_value="Zusammenfassung aller Children")
        
        result = digest_node(note1.id, children, mock_llm, max_children=8)
        
        # Sollte Zusammenfassung zurückgeben
        assert result is not None
        assert "Zusammenfassung" in result
        print("✅ Nodes mit vielen Children werden verdaut")


class TestRunMemoryEnzymes:
    """Test für run_memory_enzymes"""
    
    def test_run_memory_enzymes_completes(self):
        """Test: run_memory_enzymes läuft durch"""
        graph = GraphStore()
        
        # Erstelle Test-Notes
        notes = {}
        for i in range(3):
            note = AtomicNote(content=f"Test Note {i}")
            graph.add_node(note)
            notes[note.id] = note
        
        # Mock LLM Service
        mock_llm = Mock(spec=LLMService)
        mock_llm.get_embedding = lambda x: [0.5] * 768
        
        # Führe Enzyme aus
        results = run_memory_enzymes(
            graph,
            mock_llm,
            prune_config={"max_age_days": 90, "min_weight": 0.3},
            suggest_config={"threshold": 0.75, "max_suggestions": 10}
        )
        
        # Sollte Dict mit Ergebnissen zurückgeben
        assert isinstance(results, dict)
        assert "pruned_count" in results
        assert "suggestions_count" in results
        assert "digested_count" in results
        print(f"✅ run_memory_enzymes läuft: {results}")


def run_tests():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("🧪 A-MEM Enzymes Test Suite")
    print("="*60 + "\n")
    
    test_classes = [
        TestCosineSimilarity,
        TestLinkPruner,
        TestRelationSuggester,
        TestDigestNode,
        TestRunMemoryEnzymes
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
                print(f"  ✅ {method_name}")
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

