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
    
    def __post_init__(self):
        """Set created_at timestamp if not provided."""
        if self.created_at == 0.0:
            self.created_at = datetime.now().timestamp()
    
    def score(self, recency_weight: float = 0.1) -> float:
        """
        Compute a score for ranking entries.
        
        Args:
            recency_weight: Weight for recency bonus (higher = more recent entries favored).
            
        Returns:
            Score combining helpfulness and recency.
        """
        helpfulness = self.helpful_count - self.harmful_count
        # Simple recency bonus: newer entries get a small boost
        recency_bonus = recency_weight * self.last_seen_step
        return helpfulness + recency_bonus
    
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
    
    def get_top_k(self, domain: str, k: int = 5) -> List[PlaybookEntry]:
        """
        Get top-k entries for a domain, ranked by score.
        
        Args:
            domain: Domain name (e.g., "finance", "medical").
            k: Number of entries to return.
            
        Returns:
            List of top-k PlaybookEntry objects.
        """
        domain_entries = [e for e in self.entries if e.domain == domain]
        # Sort by score (descending)
        domain_entries.sort(key=lambda e: e.score(), reverse=True)
        return domain_entries[:k]
    
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

