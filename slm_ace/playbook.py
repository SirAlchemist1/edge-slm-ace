"""ACE-style playbook for storing and managing domain-specific strategies."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
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
    # Additional fields for working memory scoring
    success_count: int = 0  # Number of times this entry appeared in correct examples
    failure_count: int = 0  # Number of times this entry appeared in incorrect examples
    last_used_at: int = 0  # Step counter when this entry was last used
    is_generic: bool = False  # Whether this entry is generic (determined by heuristics)
    
    def __post_init__(self):
        """Set created_at timestamp if not provided."""
        if self.created_at == 0.0:
            self.created_at = datetime.now().timestamp()
        # Auto-detect if entry is generic based on heuristics
        if not hasattr(self, 'is_generic') or self.is_generic is False:
            self.is_generic = self._detect_generic()
    
    def _detect_generic(self) -> bool:
        """
        Detect if this entry is generic using simple heuristics.
        
        Returns:
            True if entry appears to be generic advice.
        """
        text_lower = self.text.lower()
        generic_phrases = [
            "think carefully",
            "consider all perspectives",
            "be thorough",
            "pay attention",
            "make sure",
            "remember to",
        ]
        
        # Check if text is very short (likely generic)
        if len(self.text.split()) < 5:
            return True
        
        # Check if text contains generic phrases
        if any(phrase in text_lower for phrase in generic_phrases):
            # Only mark as generic if it's mostly generic (few specific details)
            if len(self.text.split()) < 10:
                return True
        
        return False
    
    def score(self, recency_weight: float = 0.1, current_step: int = 0) -> float:
        """
        Compute a score for ranking entries (enhanced for working memory).
        
        Args:
            recency_weight: Weight for recency bonus (higher = more recent entries favored).
            current_step: Current step counter for recency calculation.
            
        Returns:
            Score combining correctness ratio, recency, and genericity penalty.
        """
        # Correctness ratio: success / (success + failure + 1) to avoid division by zero
        total_uses = self.success_count + self.failure_count
        if total_uses > 0:
            correctness_ratio = self.success_count / (total_uses + 1)
        else:
            correctness_ratio = 0.5  # Neutral for unused entries
        
        # Recency via exponential decay: exp(-alpha * age_in_steps)
        # Age is measured in steps since last use
        age_in_steps = max(0, current_step - self.last_used_at) if current_step > 0 else 0
        alpha = 0.1  # Decay rate (smaller = slower decay)
        recency_decay = 1.0 / (1.0 + alpha * age_in_steps)  # Simplified exponential decay
        
        # Genericity penalty
        genericity_penalty = -0.5 if self.is_generic else 0.0
        
        # Combine: correctness ratio (0-1) + recency decay (0-1) + genericity penalty
        score = correctness_ratio + recency_weight * recency_decay + genericity_penalty
        
        return score
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookEntry":
        """Create from dictionary."""
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
    
    def get_top_k(self, domain: str, k: int = 5, current_step: int = 0) -> List[PlaybookEntry]:
        """
        Get top-k entries for a domain, ranked by score.
        
        Args:
            domain: Domain name (e.g., "finance", "medical").
            k: Number of entries to return.
            current_step: Current step counter for recency calculation.
            
        Returns:
            List of top-k PlaybookEntry objects.
        """
        domain_entries = [e for e in self.entries if e.domain == domain]
        # Sort by score (descending)
        domain_entries.sort(key=lambda e: e.score(current_step=current_step), reverse=True)
        return domain_entries[:k]
    
    def score_entry(self, entry: PlaybookEntry, current_step: int = 0) -> float:
        """
        Compute a score for a playbook entry (for working memory mode).
        
        This method combines:
        - Correctness ratio (success / (success + failure + 1))
        - Recency via exponential decay
        - Genericity penalty
        
        Args:
            entry: The playbook entry to score.
            current_step: Current step counter for recency calculation.
            
        Returns:
            Score as a float (higher is better).
        """
        return entry.score(current_step=current_step)
    
    def get_top_entries_for_budget(
        self,
        domain: str,
        token_budget: int,
        current_step: int = 0,
        tokens_per_word: float = 1.3,
    ) -> List[PlaybookEntry]:
        """
        Get top entries for a domain that fit within a token budget.
        
        Entries are sorted by score (descending) and greedily included
        until the estimated total token count <= token_budget.
        
        Args:
            domain: Domain name (e.g., "finance", "medical").
            token_budget: Maximum number of tokens allowed.
            current_step: Current step counter for recency calculation.
            tokens_per_word: Estimated tokens per word (default: 1.3).
            
        Returns:
            List of PlaybookEntry objects that fit within the budget.
        """
        domain_entries = [e for e in self.entries if e.domain == domain]
        
        # Sort by score (descending)
        domain_entries.sort(key=lambda e: e.score(current_step=current_step), reverse=True)
        
        # Greedily include entries until budget is exceeded
        selected_entries = []
        total_tokens = 0
        
        for entry in domain_entries:
            # Estimate token count: approximate via word count
            word_count = len(entry.text.split())
            estimated_tokens = int(word_count * tokens_per_word)
            
            if total_tokens + estimated_tokens <= token_budget:
                selected_entries.append(entry)
                total_tokens += estimated_tokens
            else:
                # Can't fit this entry, stop
                break
        
        return selected_entries
    
    def add_entry(
        self,
        domain: str,
        text: str,
        step: int,
        similarity_threshold: float = 0.8,
    ) -> PlaybookEntry:
        """
        Add a new entry to the playbook, with deduplication.
        
        Args:
            domain: Domain name.
            text: Strategy text.
            step: Current step/index.
            similarity_threshold: Threshold for considering entries similar (simple containment check).
            
        Returns:
            The added or existing PlaybookEntry.
        """
        # Simple deduplication: check if a similar entry exists
        text_lower = text.lower().strip()
        for entry in self.entries:
            if entry.domain == domain:
                entry_text_lower = entry.text.lower().strip()
                # Check if one contains the other (simple similarity)
                if text_lower in entry_text_lower or entry_text_lower in text_lower:
                    # Update last_seen_step
                    entry.last_seen_step = step
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
            success_count=0,
            failure_count=0,
            last_used_at=step,
            is_generic=False,  # Will be auto-detected in __post_init__
        )
        self.entries.append(entry)
        return entry
    
    def record_feedback(self, entry_id: str, helpful: bool) -> None:
        """
        Record feedback for an entry (helpful or harmful).
        
        Also updates success_count and failure_count for working memory scoring.
        
        Args:
            entry_id: ID of the entry.
            helpful: True if helpful, False if harmful.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                if helpful:
                    entry.helpful_count += 1
                    entry.success_count += 1
                else:
                    entry.harmful_count += 1
                    entry.failure_count += 1
                return
        
        # Entry not found - could raise an error, but silently ignore for robustness
        pass
    
    def mark_entry_used(self, entry_id: str, step: int) -> None:
        """
        Mark an entry as used at a specific step (for recency tracking).
        
        Args:
            entry_id: ID of the entry.
            step: Current step counter.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                entry.last_used_at = step
                entry.last_seen_step = step
                return
    
    def prune(self, max_entries_per_domain: int = 32) -> None:
        """
        Prune playbook to keep only top entries per domain.
        
        Args:
            max_entries_per_domain: Maximum entries to keep per domain.
        """
        domains = set(e.domain for e in self.entries)
        pruned_entries = []
        
        for domain in domains:
            domain_entries = [e for e in self.entries if e.domain == domain]
            # Sort by score (descending)
            domain_entries.sort(key=lambda e: e.score(), reverse=True)
            # Keep top-k
            pruned_entries.extend(domain_entries[:max_entries_per_domain])
        
        self.entries = pruned_entries

