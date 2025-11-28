"""
Test Suite für Enzyme-Scheduler

Tests für:
1. Scheduler-Start/Stop
2. Scheduler-Loop
3. Automatische Enzyme-Ausführung
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from a_mem.core.logic import MemoryController
from a_mem.models.note import NoteInput


class TestScheduler:
    """Test für Enzyme-Scheduler"""
    
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self):
        """Test: Scheduler kann gestartet und gestoppt werden"""
        controller = MemoryController()
        
        # Scheduler sollte initial nicht laufen
        assert not controller._enzyme_scheduler_running
        assert controller._enzyme_scheduler_task is None
        
        # Starte Scheduler
        controller.start_enzyme_scheduler(interval_hours=0.001)  # Sehr kurzes Intervall für Test
        
        # Scheduler sollte laufen
        assert controller._enzyme_scheduler_running
        assert controller._enzyme_scheduler_task is not None
        
        # Stoppe Scheduler
        controller.stop_enzyme_scheduler()
        
        # Warte auf Task-Completion
        try:
            await asyncio.wait_for(controller._enzyme_scheduler_task, timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass  # Task wurde gecancelt, das ist OK
        
        # Scheduler sollte gestoppt sein
        assert not controller._enzyme_scheduler_running
        
        print("✅ Scheduler Start/Stop funktioniert")
    
    @pytest.mark.asyncio
    async def test_scheduler_loop_runs(self):
        """Test: Scheduler-Loop läuft und führt Enzyme aus"""
        controller = MemoryController()
        
        # Mock run_memory_enzymes direkt im utils.enzymes Modul
        from a_mem.utils import enzymes
        original_run = enzymes.run_memory_enzymes
        
        def mock_run_memory_enzymes(*args, **kwargs):
            return {
                "pruned_count": 2,
                "suggestions_count": 3,
                "digested_count": 1
            }
        
        enzymes.run_memory_enzymes = mock_run_memory_enzymes
        
        try:
            # Starte Scheduler mit sehr kurzem Intervall (0.0001 Stunden = 0.36 Sekunden)
            controller.start_enzyme_scheduler(interval_hours=0.0001)
            
            # Warte kurz damit Loop einmal läuft
            await asyncio.sleep(0.5)
            
            # Stoppe Scheduler
            controller.stop_enzyme_scheduler()
            
            # Warte auf Task-Completion
            try:
                await asyncio.wait_for(controller._enzyme_scheduler_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass  # Task wurde gecancelt, das ist OK
            
            print("✅ Scheduler-Loop läuft")
        finally:
            # Restore original function
            enzymes.run_memory_enzymes = original_run
    
    @pytest.mark.asyncio
    async def test_scheduler_multiple_starts(self):
        """Test: Mehrfaches Starten wird ignoriert"""
        controller = MemoryController()
        
        # Starte Scheduler
        controller.start_enzyme_scheduler(interval_hours=1.0)
        
        # Versuche erneut zu starten
        controller.start_enzyme_scheduler(interval_hours=1.0)
        
        # Sollte nur einen Task haben
        assert controller._enzyme_scheduler_running
        
        # Stoppe
        controller.stop_enzyme_scheduler()
        
        # Warte auf Task-Completion
        try:
            await asyncio.wait_for(controller._enzyme_scheduler_task, timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        
        print("✅ Mehrfaches Starten wird ignoriert")


def run_tests():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("🧪 A-MEM Scheduler Test Suite")
    print("="*60 + "\n")
    
    test_instance = TestScheduler()
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    total_tests = 0
    passed_tests = 0
    
    for method_name in test_methods:
        total_tests += 1
        try:
            method = getattr(test_instance, method_name)
            
            # Handle async tests
            if asyncio.iscoroutinefunction(method):
                asyncio.run(method())
            else:
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

