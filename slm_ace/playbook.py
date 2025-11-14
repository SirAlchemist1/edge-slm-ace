"""ACE-style playbook for storing and managing domain-specific strategies."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime


@dataclass
class PlaybookEntry:
    """A single entry in the ACE playbook."""
    id: str
    domain: str
    text: str
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: float = 0.0
    last_seen_step: int = 0
    task_type: Optional[str] = None  # e.g., "calculation", "extraction", "reasoning"
    keywords: Optional[List[str]] = None  # Extracted keywords for semantic matching
    
    def __post_init__(self):
        """Set created_at timestamp if not provided."""
        if self.created_at == 0.0:
            self.created_at = datetime.now().timestamp()
        # Extract keywords if not provided
        if self.keywords is None:
            self.keywords = self._extract_keywords()
        # Infer task type if not provided
        if self.task_type is None:
            self.task_type = self._infer_task_type()
    
    def _extract_keywords(self) -> List[str]:
        """Extract important keywords from the strategy text."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "may", "might", "must", "can", "this", "that", "these", "those"
        }
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', self.text.lower())
        # Filter stop words and return unique keywords
        keywords = [w for w in words if w not in stop_words]
        # Return top 10 most frequent or all if less than 10
        return list(set(keywords))[:10]
    
    def _infer_task_type(self) -> str:
        """Infer task type from strategy text."""
        text_lower = self.text.lower()
        # Finance-specific patterns
        if any(word in text_lower for word in ["calculate", "compute", "formula", "subtract", "add", "multiply", "divide"]):
            return "calculation"
        elif any(word in text_lower for word in ["extract", "find", "identify", "locate", "search"]):
            return "extraction"
        elif any(word in text_lower for word in ["reason", "step", "process", "approach", "method"]):
            return "reasoning"
        else:
            return "general"
    
    def score(self, recency_weight: float = 0.1, decay_factor: float = 0.95) -> float:
        """
        Compute a score for ranking entries with smart forgetting.
        
        Args:
            recency_weight: Weight for recency bonus (higher = more recent entries favored).
            decay_factor: Decay factor for older entries (0.95 = 5% decay per step).
            
        Returns:
            Score combining helpfulness, recency, and age decay.
        """
        helpfulness = self.helpful_count - self.harmful_count
        
        # Recency bonus: more recent entries get boost
        recency_bonus = recency_weight * self.last_seen_step
        
        # Age decay: older entries (by creation time) get slight penalty
        # This implements "forgetful in the right way" - very old, low-value entries fade
        age_penalty = 0.0
        if helpfulness <= 0:  # Only penalize unhelpful or neutral entries
            # Older entries with no positive feedback get penalized more
            age_penalty = -0.5
        
        # Quality score: entries with high helpfulness ratio are preferred
        total_feedback = self.helpful_count + self.harmful_count
        quality_bonus = 0.0
        if total_feedback > 0:
            helpful_ratio = self.helpful_count / total_feedback
            # Bonus for entries with >70% helpful ratio
            if helpful_ratio > 0.7:
                quality_bonus = helpful_ratio * 2.0
        
        return helpfulness + recency_bonus + age_penalty + quality_bonus
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookEntry":
        """Create from dictionary with backward compatibility."""
        # Handle old playbooks that don't have task_type and keywords
        if "task_type" not in d:
            d["task_type"] = None
        if "keywords" not in d:
            d["keywords"] = None
        return cls(**d)


class Playbook:
    """ACE-style playbook storing domain-specific strategies."""
    
    def __init__(self, entries: Optional[List[PlaybookEntry]] = None):
        """Initialize playbook with optional entries."""
        self.entries: List[PlaybookEntry] = entries or []
        self._next_id = 1
    
    @classmethod
    def load(cls, path: Path) -> "Playbook":
        """
        Load playbook from a JSONL file.
        
        Args:
            path: Path to JSONL file.
            
        Returns:
            Playbook instance.
        """
        entries = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry_dict = json.loads(line)
                        entries.append(PlaybookEntry.from_dict(entry_dict))
        
        playbook = cls(entries)
        # Set next_id to max existing id + 1
        if entries:
            max_id = max(int(e.id) for e in entries if e.id.isdigit())
            playbook._next_id = max_id + 1
        return playbook
    
    def save(self, path: Path) -> None:
        """
        Save playbook to a JSONL file.
        
        Args:
            path: Path to save JSONL file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using Jaccard similarity on keywords.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Extract keywords from both texts
        def get_keywords(t: str) -> Set[str]:
            stop_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "is", "are", "was", "were", "be", "been"
            }
            words = re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
            return set(w for w in words if w not in stop_words)
        
        keywords1 = get_keywords(text1)
        keywords2 = get_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_top_k(
        self,
        domain: str,
        k: int = 5,
        question: Optional[str] = None,
        context: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[PlaybookEntry]:
        """
        Get top-k entries for a domain, ranked by score and relevance.
        
        Args:
            domain: Domain name (e.g., "finance", "medical").
            k: Number of entries to return.
            question: Optional question text for relevance matching.
            context: Optional context text for relevance matching.
            task_type: Optional task type filter (e.g., "calculation", "extraction").
            
        Returns:
            List of top-k PlaybookEntry objects, sorted by relevance and score.
        """
        domain_entries = [e for e in self.entries if e.domain == domain]
        
        # Filter by task type if specified
        if task_type:
            domain_entries = [e for e in domain_entries if e.task_type == task_type]
        
        # If question/context provided, compute relevance scores
        if question or context:
            query_text = f"{question or ''} {context or ''}".strip()
            for entry in domain_entries:
                # Compute semantic similarity
                similarity = self._compute_semantic_similarity(query_text, entry.text)
                # Boost score by similarity
                entry._relevance_boost = similarity * 2.0  # Temporary attribute
            # Sort by (score + relevance_boost)
            domain_entries.sort(
                key=lambda e: e.score() + getattr(e, '_relevance_boost', 0.0),
                reverse=True
            )
            # Clean up temporary attribute
            for entry in domain_entries:
                if hasattr(entry, '_relevance_boost'):
                    delattr(entry, '_relevance_boost')
        else:
            # Just sort by score
            domain_entries.sort(key=lambda e: e.score(), reverse=True)
        
        return domain_entries[:k]
    
    def add_entry(
        self,
        domain: str,
        text: str,
        step: int,
        similarity_threshold: float = 0.6,
        task_type: Optional[str] = None,
    ) -> PlaybookEntry:
        """
        Add a new entry to the playbook, with semantic deduplication.
        
        Args:
            domain: Domain name.
            text: Strategy text.
            step: Current step/index.
            similarity_threshold: Threshold for considering entries similar (0.0 to 1.0).
            task_type: Optional task type (e.g., "calculation", "extraction").
            
        Returns:
            The added or existing PlaybookEntry.
        """
        text = text.strip()
        if not text:
            raise ValueError("Cannot add empty entry")
        
        # Check for semantic similarity with existing entries
        for entry in self.entries:
            if entry.domain == domain:
                similarity = self._compute_semantic_similarity(text, entry.text)
                # Also check for containment (for very similar entries)
                text_lower = text.lower()
                entry_lower = entry.text.lower()
                containment_match = (
                    text_lower in entry_lower or entry_lower in text_lower
                ) and len(text_lower) > 20  # Only for substantial texts
                
                if similarity >= similarity_threshold or containment_match:
                    # Update last_seen_step to mark as recently used
                    entry.last_seen_step = step
                    # If new text is more specific/detailed, update entry text
                    if len(text) > len(entry.text) * 1.2:
                        entry.text = text
                        entry.keywords = None  # Reset to recompute
                        # Re-infer task type if not provided
                        if task_type:
                            entry.task_type = task_type
                        else:
                            # Create temporary entry to infer type
                            temp_entry = PlaybookEntry(id="temp", domain=domain, text=text)
                            entry.task_type = temp_entry.task_type
                    return entry
        
        # Create new entry
        entry_id = str(self._next_id)
        self._next_id += 1
        
        entry = PlaybookEntry(
            id=entry_id,
            domain=domain,
            text=text,
            helpful_count=0,
            harmful_count=0,
            last_seen_step=step,
            task_type=task_type,
        )
        self.entries.append(entry)
        return entry
    
    def record_feedback(self, entry_id: str, helpful: bool) -> None:
        """
        Record feedback for an entry (helpful or harmful).
        
        Args:
            entry_id: ID of the entry.
            helpful: True if helpful, False if harmful.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                if helpful:
                    entry.helpful_count += 1
                else:
                    entry.harmful_count += 1
                return
        
        # Entry not found - could raise an error, but silently ignore for robustness
        pass
    
    def prune(
        self,
        max_entries_per_domain: int = 32,
        min_helpfulness: int = -2,
        keep_recent_n: int = 5,
    ) -> None:
        """
        Prune playbook with smart forgetting - keeps valuable entries, removes low-value ones.
        
        Smart forgetting strategy:
        - Always keep top-scoring entries
        - Always keep recent entries (even if low score) for exploration
        - Remove entries with very negative helpfulness scores
        - Prefer entries with high helpfulness ratio
        
        Args:
            max_entries_per_domain: Maximum entries to keep per domain.
            min_helpfulness: Minimum helpfulness score (helpful_count - harmful_count) to keep.
            keep_recent_n: Always keep N most recent entries even if low-scoring.
        """
        domains = set(e.domain for e in self.entries)
        pruned_entries = []
        
        for domain in domains:
            domain_entries = [e for e in self.entries if e.domain == domain]
            
            # Sort by score (descending)
            domain_entries.sort(key=lambda e: e.score(), reverse=True)
            
            # Always keep top-scoring entries
            top_entries = domain_entries[:max_entries_per_domain]
            
            # Also keep recent entries (for exploration) that aren't already in top
            recent_entries = sorted(
                domain_entries,
                key=lambda e: e.last_seen_step,
                reverse=True
            )[:keep_recent_n]
            
            # Combine and deduplicate
            kept_entries = {}
            for entry in top_entries + recent_entries:
                if entry.id not in kept_entries:
                    # Only add if meets minimum helpfulness or is very recent
                    helpfulness = entry.helpful_count - entry.harmful_count
                    is_recent = entry.last_seen_step > 0  # Has been seen recently
                    if helpfulness >= min_helpfulness or is_recent:
                        kept_entries[entry.id] = entry
            
            pruned_entries.extend(kept_entries.values())
        
        self.entries = pruned_entries

