"""
Core Logic: MemoryController

Implements Async Non-Blocking I/O using `run_in_executor` and Batch-Saving strategy.
"""

import asyncio
from typing import Any, Dict, List, Optional
from ..storage.engine import StorageManager
from ..utils.llm import LLMService
from ..models.note import AtomicNote, NoteInput, NoteRelation, SearchResult
from ..utils.serializers import serialize_note
from ..utils.priority import log_event, compute_priority
from ..utils.enzymes import run_memory_enzymes

class MemoryController:
    def __init__(self):
        self.storage = StorageManager()
        self.llm = LLMService()
        self._enzyme_scheduler_task = None
        self._enzyme_scheduler_running = False

    async def create_note(self, input_data: NoteInput) -> str:
        """
        Phase 1: Creation. 
        Critical I/O operations are offloaded to threads.
        """
        loop = asyncio.get_running_loop()

        # 1. CPU-Bound / Network Ops (LLM)
        # LLM calls sind meist intern async oder schnell genug, hier sync wrapper
        metadata = await loop.run_in_executor(None, self.llm.extract_metadata, input_data.content)
        
        # 2. Objekt Erstellung
        note = AtomicNote(
            content=input_data.content,
            contextual_summary=metadata.get("summary", ""),
            keywords=metadata.get("keywords", []),
            tags=metadata.get("tags", []),
            type=metadata.get("type")  # Optional: rule, procedure, concept, tool, reference, integration
        )
        
        # 3. Embedding calculation (Paper Section 3.1, Formula 3):
        # ei = fenc[concat(ci, Ki, Gi, Xi)]
        # Concatenation of all text components for complete semantic representation
        text_for_embedding = f"{note.content} {note.contextual_summary} {' '.join(note.keywords)} {' '.join(note.tags)}"
        embedding = await loop.run_in_executor(None, self.llm.get_embedding, text_for_embedding)
        
        # 4. Blocking I/O Offloading (Storage)
        await loop.run_in_executor(None, self.storage.vector.add, note, embedding)
        await loop.run_in_executor(None, self.storage.graph.add_node, note)
        
        # Explicit snapshot save after adding
        await loop.run_in_executor(None, self.storage.graph.save_snapshot)
        
        # 5. Event Logging
        log_event("NOTE_CREATED", {
            "id": note.id,
            "type": note.type,
            "tags": note.tags,
            "keywords_count": len(note.keywords)
        })
        
        # 6. Background Evolution
        asyncio.create_task(self._evolve_memory(note, embedding))
        
        return note.id

    async def _evolve_memory(self, new_note: AtomicNote, embedding: List[float]):
        """
        Phase 2: Asynchronous Knowledge Evolution.
        Batch-Update strategy for the graph.
        """
        loop = asyncio.get_running_loop()
        print(f"🔄 Evolving memory for note {new_note.id}...")
        
        try:
            # 1. Candidate search (I/O in thread)
            candidate_ids, distances = await loop.run_in_executor(
                None, self.storage.vector.query, embedding, 5
            )
            
            links_found = 0
            evolutions_found = 0
            candidate_notes = []
            
            # 2. Linking logic + Memory Evolution
            for c_id, dist in zip(candidate_ids, distances):
                if c_id == new_note.id: continue
                
                candidate_note = self.storage.get_note(c_id)
                if not candidate_note: continue
                
                candidate_notes.append(candidate_note)

                # LLM Check (Network I/O)
                # In Production sollte check_link auch async sein, hier wrapper
                is_related, relation = await loop.run_in_executor(
                    None, self.llm.check_link, new_note, candidate_note
                )
                
                if is_related and relation:
                    print(f"🔗 Linking {new_note.id} -> {c_id} ({relation.relation_type})")
                    # In-Memory Update (fast)
                    self.storage.graph.add_edge(relation)
                    links_found += 1
                    # Event Logging
                    log_event("RELATION_CREATED", {
                        "from": new_note.id,
                        "to": c_id,
                        "relation_type": relation.relation_type,
                        "reasoning": relation.reasoning
                    })
            
            # 3. Memory Evolution (Paper Section 3.3)
            # Check if existing memories should be updated
            for candidate_note in candidate_notes:
                evolved_note = await loop.run_in_executor(
                    None, self.llm.evolve_memory, new_note, candidate_note
                )
                
                if evolved_note:
                    print(f"🧠 Evolving memory {candidate_note.id} based on new information")
                    
                    # Calculate new embedding (Paper Section 3.1, Formula 3):
                    # ei = fenc[concat(ci, Ki, Gi, Xi)]
                    # Concatenation of all text components including tags
                    evolved_text = f"{evolved_note.content} {evolved_note.contextual_summary} {' '.join(evolved_note.keywords)} {' '.join(evolved_note.tags)}"
                    new_embedding = await loop.run_in_executor(
                        None, self.llm.get_embedding, evolved_text
                    )
                    
                    # Update in VectorStore
                    await loop.run_in_executor(
                        None, self.storage.vector.update, 
                        candidate_note.id, evolved_note, new_embedding
                    )
                    
                    # Update in GraphStore
                    await loop.run_in_executor(
                        None, self.storage.graph.update_node, evolved_note
                    )
                    
                    evolutions_found += 1
                    # Event Logging
                    log_event("MEMORY_EVOLVED", {
                        "note_id": candidate_note.id,
                        "triggered_by": new_note.id,
                        "updated_fields": ["summary", "keywords", "tags"]
                    })
            
            # 4. Batch Save (Single write to disk)
            if links_found > 0 or evolutions_found > 0:
                await loop.run_in_executor(None, self.storage.graph.save_snapshot)
                print(f"✅ Evolution finished. {links_found} links, {evolutions_found} memory updates saved.")
            else:
                print("✅ Evolution finished. No new links or updates.")

        except Exception as e:
            print(f"❌ Evolution failed for {new_note.id}: {e}")

    async def retrieve(self, query: str) -> List[SearchResult]:
        loop = asyncio.get_running_loop()
        
        # Embedding calculation
        q_embedding = await loop.run_in_executor(None, self.llm.get_embedding, query)
        
        # Vector Query
        ids, scores = await loop.run_in_executor(None, self.storage.vector.query, q_embedding, 5)
        
        results = []
        for n_id, similarity_score in zip(ids, scores):
            note = self.storage.get_note(n_id)
            if not note: continue
            
            # Get edge count (graph degree) for priority calculation
            graph = self.storage.graph.graph
            edge_count = graph.degree(n_id) if n_id in graph else 0
            
            # Compute priority score (on-the-fly)
            priority = compute_priority(note, usage_count=0, edge_count=edge_count)
            
            # Combined score: similarity * priority (both normalized)
            # Similarity is already 0-1, priority is typically 0.3-2.0, so we normalize it
            normalized_priority = min(priority / 2.0, 1.0)  # Cap at 1.0
            combined_score = similarity_score * (0.7 + 0.3 * normalized_priority)
            
            # Graph Traversal (In-Memory, fast enough for Main Thread)
            neighbors_data = self.storage.graph.get_neighbors(n_id)
            related_notes = []
            for n in neighbors_data:
                # Validate and filter invalid nodes
                if not n or not isinstance(n, dict):
                    continue
                # Check if content is present (required field)
                if "content" not in n or not n.get("content"):
                    continue
                try:
                    related_note = AtomicNote(**n)
                    related_notes.append(related_note)
                except Exception as e:
                    # Skip invalid nodes (e.g., corrupted by evolution)
                    print(f"Warning: Skipping invalid neighbor node: {e}")
                    continue
            
            results.append(SearchResult(
                note=note,
                score=combined_score,  # Use combined score instead of raw similarity
                related_notes=related_notes
            ))
        
        # Sort by combined score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    async def delete_note(self, note_id: str) -> bool:
        """Deletes a note from Graph and Vector Store."""
        loop = asyncio.get_running_loop()
        
        # Check if note exists in graph (directly, not via get_note)
        note_exists = await loop.run_in_executor(
            None, lambda: note_id in self.storage.graph.graph
        )
        if not note_exists:
            return False
        
        # Delete from both stores (in thread)
        success = await loop.run_in_executor(
            None, self.storage.delete_note, note_id
        )
        
        if success:
            # Save graph snapshot after deletion
            await loop.run_in_executor(None, self.storage.graph.save_snapshot)
        
        return success
    
    async def reset_memory(self) -> bool:
        """Resets the complete memory system (Graph + Vector Store)."""
        loop = asyncio.get_running_loop()
        
        try:
            # Reset in Thread (blocking I/O)
            await loop.run_in_executor(None, self.storage.reset)
            return True
        except Exception as e:
            print(f"Error resetting memory: {e}")
            return False

    async def list_notes_data(self) -> List[Dict[str, Any]]:
        """Returns all notes as JSON-serializable dicts."""
        loop = asyncio.get_running_loop()

        def _collect():
            notes = []
            for node_id, attrs in self.storage.graph.graph.nodes(data=True):
                data = dict(attrs)
                data.setdefault("id", node_id)
                try:
                    note = AtomicNote(**data)
                    notes.append(serialize_note(note))
                except Exception as exc:
                    notes.append({"id": node_id, "error": str(exc)})
            return notes

        return await loop.run_in_executor(None, _collect)

    async def get_note_data(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Returns a single note as JSON-serializable dict."""
        loop = asyncio.get_running_loop()

        def _get():
            note = self.storage.get_note(note_id)
            if not note:
                return None
            return serialize_note(note)

        return await loop.run_in_executor(None, _get)

    async def list_relations_data(self, note_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns relations, optionally filtered by note id."""
        loop = asyncio.get_running_loop()

        def _collect():
            edges = []
            for source, target, attrs in self.storage.graph.graph.edges(data=True):
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
            return edges

        return await loop.run_in_executor(None, _collect)

    async def get_graph_snapshot(self) -> Dict[str, Any]:
        """Returns nodes + edges for visualization."""
        loop = asyncio.get_running_loop()

        def _snapshot():
            nodes = []
            for node_id, attrs in self.storage.graph.graph.nodes(data=True):
                data = dict(attrs)
                data.setdefault("id", node_id)
                nodes.append(data)

            edges = []
            for source, target, attrs in self.storage.graph.graph.edges(data=True):
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

        return await loop.run_in_executor(None, _snapshot)

    async def update_note_metadata(self, note_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Updates contextual summary/tags/keywords for a note."""
        loop = asyncio.get_running_loop()

        def _update():
            note = self.storage.get_note(note_id)
            if not note:
                return {"error": f"Note '{note_id}' not found"}

            if "content" in payload:
                return {"error": "Content edits require re-embedding and are not supported."}

            allowed = {"contextual_summary", "tags", "keywords"}
            updates = {k: v for k, v in payload.items() if k in allowed}
            if not updates:
                return {"error": "No valid fields provided"}

            updated = note.model_copy(update=updates)
            self.storage.graph.update_node(updated)
            self.storage.graph.save_snapshot()
            return {"note": serialize_note(updated)}

        return await loop.run_in_executor(None, _update)

    async def delete_note_data(self, note_id: str) -> Dict[str, Any]:
        """Deletes a note and returns status payload."""
        success = await self.delete_note(note_id)
        if success:
            return {
                "status": "success",
                "note_id": note_id,
                "message": f"Note {note_id} deleted successfully. All connections removed.",
            }
        return {
            "status": "error",
            "note_id": note_id,
            "message": f"Note {note_id} not found or could not be deleted.",
        }

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "relates_to",
        reasoning: str = "Manual link",
        weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Adds a manual relation between two notes."""
        loop = asyncio.get_running_loop()

        def _add():
            graph = self.storage.graph.graph
            if source_id not in graph.nodes:
                return {"error": f"Source '{source_id}' not found"}
            if target_id not in graph.nodes:
                return {"error": f"Target '{target_id}' not found"}

            relation = NoteRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                reasoning=reasoning,
                weight=weight,
            )
            self.storage.graph.add_edge(relation)
            self.storage.graph.save_snapshot()
            return {"status": "edge_added", "relation": relation.model_dump()}

        return await loop.run_in_executor(None, _add)

    async def remove_relation(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Removes a relation between two nodes."""
        loop = asyncio.get_running_loop()

        def _remove():
            graph = self.storage.graph.graph
            if not graph.has_edge(source_id, target_id):
                return {"error": f"Relation {source_id}->{target_id} not found"}
            graph.remove_edge(source_id, target_id)
            self.storage.graph.save_snapshot()
            return {"status": "edge_removed", "source": source_id, "target": target_id}

        return await loop.run_in_executor(None, _remove)
    
    def start_enzyme_scheduler(self, interval_hours: float = 1.0):
        """
        Startet den automatischen Enzyme-Scheduler.
        
        Args:
            interval_hours: Intervall in Stunden (default: 1.0 = 1 Stunde)
        """
        if self._enzyme_scheduler_running:
            print("⚠️ Enzyme-Scheduler läuft bereits")
            return
        
        self._enzyme_scheduler_running = True
        self._enzyme_scheduler_task = asyncio.create_task(
            self._enzyme_scheduler_loop(interval_hours)
        )
        print(f"✅ Enzyme-Scheduler gestartet (Intervall: {interval_hours}h)")
        log_event("ENZYME_SCHEDULER_STARTED", {
            "interval_hours": interval_hours
        })
    
    def stop_enzyme_scheduler(self):
        """Stoppt den automatischen Enzyme-Scheduler."""
        if self._enzyme_scheduler_task:
            self._enzyme_scheduler_task.cancel()
            self._enzyme_scheduler_running = False
            print("🛑 Enzyme-Scheduler gestoppt")
            log_event("ENZYME_SCHEDULER_STOPPED", {})
    
    async def _enzyme_scheduler_loop(self, interval_hours: float):
        """
        Background-Loop für automatische Enzyme-Ausführung.
        
        Args:
            interval_hours: Intervall in Stunden
        """
        interval_seconds = interval_hours * 3600
        
        while self._enzyme_scheduler_running:
            try:
                # Warte auf Intervall
                await asyncio.sleep(interval_seconds)
                
                # Führe Enzyme aus
                print(f"🔄 [Scheduler] Führe Memory-Enzyme aus...")
                loop = asyncio.get_running_loop()
                
                def _run_enzymes():
                    return run_memory_enzymes(
                        self.storage.graph,
                        self.llm,
                        prune_config={
                            "max_age_days": 90,
                            "min_weight": 0.3
                        },
                        suggest_config={
                            "threshold": 0.75,
                            "max_suggestions": 10
                        }
                    )
                
                results = await loop.run_in_executor(None, _run_enzymes)
                
                # Graph speichern
                await loop.run_in_executor(None, self.storage.graph.save_snapshot)
                
                print(f"✅ [Scheduler] Enzyme abgeschlossen: {results['pruned_count']} pruned, {results['suggestions_count']} suggested, {results['digested_count']} digested")
                
                log_event("ENZYME_SCHEDULER_RUN", {
                    "results": results,
                    "interval_hours": interval_hours
                })
                
            except asyncio.CancelledError:
                print("🛑 [Scheduler] Wurde gestoppt")
                break
            except Exception as e:
                print(f"❌ [Scheduler] Fehler bei Enzyme-Ausführung: {e}")
                log_event("ENZYME_SCHEDULER_ERROR", {
                    "error": str(e)
                })
                # Warte kurz bevor Retry (um nicht in Endlosschleife zu kommen)
                await asyncio.sleep(60)  # 1 Minute

