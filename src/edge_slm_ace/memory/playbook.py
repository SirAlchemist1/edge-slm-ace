"""ACE-style playbook for storing and managing domain-specific strategies.

This module implements the TinyACE working memory system with:
- Retention scoring based on success/failure rates, recency, and vagueness
- Token-budgeted eviction (strategic forgetting)
- Support for both ace_full (top-k) and ace_working_memory modes
"""

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime


# Default hyperparameters for retention scoring
# These match the formal equation:
# S(l_i, t) = α·(N_succ/N_used+ε) - β·(N_fail/N_used+ε) + γ·exp(-λ·(t-t_last)) - δ·V(l_i)
DEFAULT_ALPHA = 1.0      # Weight for success ratio
DEFAULT_BETA = 0.5       # Weight for failure ratio (penalty)
DEFAULT_GAMMA = 0.3      # Weight for recency bonus
DEFAULT_DELTA = 0.4      # Weight for vagueness penalty
DEFAULT_LAMBDA = 0.05    # Decay rate for recency (smaller = slower decay)
DEFAULT_EPSILON = 1.0    # Smoothing constant to avoid division by zero


@dataclass
class ScoringParams:
    """Hyperparameters for the retention scoring formula."""
    alpha: float = DEFAULT_ALPHA      # Success ratio weight
    beta: float = DEFAULT_BETA        # Failure ratio weight (penalty)
    gamma: float = DEFAULT_GAMMA      # Recency bonus weight
    delta: float = DEFAULT_DELTA      # Vagueness penalty weight
    lambda_decay: float = DEFAULT_LAMBDA  # Recency decay rate
    epsilon: float = DEFAULT_EPSILON  # Smoothing constant
    # Ablation flags
    disable_vagueness_penalty: bool = False  # If True, set δ=0 (ignore vagueness term)
    disable_recency_decay: bool = False      # If True, set γ=0 (ignore recency term)
    disable_failure_penalty: bool = False   # If True, set β=0 (ignore failure term)
    fifo_memory: bool = False                # If True, bypass scoring entirely, use FIFO eviction


# Generic phrases that indicate vague/unhelpful lessons
GENERIC_PHRASES = [
    "think carefully",
    "think step by step",
    "consider all perspectives",
    "be thorough",
    "pay attention",
    "make sure",
    "remember to",
    "check your work",
    "read carefully",
    "double check",
    "be careful",
    "take your time",
]


def compute_vagueness_score(text: str) -> float:
    """
    Compute a vagueness/genericness score for a lesson.
    
    Returns a score in [0, 1] where:
    - 0.0 = specific, actionable lesson
    - 1.0 = very vague/generic lesson
    
    Heuristics used:
    - Presence of generic phrases
    - Very short text (< 5 words)
    - Low specificity (no numbers, formulas, or domain terms)
    
    Args:
        text: The lesson text to score.
        
    Returns:
        Vagueness score between 0 and 1.
    """
    text_lower = text.lower().strip()
    word_count = len(text.split())
    
    score = 0.0
    
    # Very short text is likely generic
    if word_count < 5:
        score += 0.5
    elif word_count < 10:
        score += 0.2
    
    # Check for generic phrases
    generic_count = sum(1 for phrase in GENERIC_PHRASES if phrase in text_lower)
    if generic_count > 0:
        # More generic phrases = higher vagueness
        score += min(0.5, generic_count * 0.2)
    
    # Check for specificity indicators (formulas, numbers, specific terms)
    has_numbers = any(c.isdigit() for c in text)
    has_formula = any(sym in text for sym in ['=', '+', '-', '*', '/', '%', '^'])
    has_specific_terms = any(term in text_lower for term in [
        'formula', 'equation', 'calculate', 'multiply', 'divide', 
        'subtract', 'add', 'percentage', 'ratio', 'if', 'when', 'then'
    ])
    
    # Reduce score for specific content
    if has_numbers:
        score -= 0.15
    if has_formula:
        score -= 0.15
    if has_specific_terms:
        score -= 0.1
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


@dataclass
class PlaybookEntry:
    """A single entry in the ACE playbook.
    
    Attributes:
        id: Unique identifier for this entry.
        domain: Domain/task this lesson applies to (e.g., "finance", "medical").
        text: The lesson text.
        success_count: Number of times this entry was used and the answer was correct.
        failure_count: Number of times this entry was used and the answer was incorrect.
        created_at: Timestamp when this entry was created.
        last_used_at: Step counter when this entry was last used in a prompt.
        token_count: Estimated token count for this entry's text.
        vagueness_score: Cached vagueness score (0=specific, 1=vague).
    """
    id: str
    domain: str
    text: str
    success_count: int = 0
    failure_count: int = 0
    created_at: float = 0.0
    last_used_at: int = 0
    token_count: int = 0
    vagueness_score: float = 0.0
    # Legacy fields for backward compatibility
    helpful_count: int = 0
    harmful_count: int = 0
    last_seen_step: int = 0
    is_generic: bool = False
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.created_at == 0.0:
            self.created_at = datetime.now().timestamp()
        
        # Compute token count if not set
        if self.token_count == 0:
            self.token_count = self._estimate_tokens()
        
        # Compute vagueness score if not set
        if self.vagueness_score == 0.0:
            self.vagueness_score = compute_vagueness_score(self.text)
        
        # Update legacy field
        self.is_generic = self.vagueness_score > 0.5
    
    def _estimate_tokens(self, tokens_per_word: float = 1.3) -> int:
        """Estimate token count from word count."""
        word_count = len(self.text.split())
        return max(1, int(word_count * tokens_per_word))
    
    def total_uses(self) -> int:
        """Return total number of times this entry was used."""
        return self.success_count + self.failure_count
    
    def score(
        self,
        current_step: int = 0,
        params: Optional[ScoringParams] = None,
    ) -> float:
        """
        Compute the retention score for this entry.
        
        Formula:
        S(l_i, t) = α·(N_succ/(N_used+ε)) - β·(N_fail/(N_used+ε)) 
                  + γ·exp(-λ·(t - t_last)) - δ·V(l_i)
        
        Ablation flags can disable individual terms:
        - disable_vagueness_penalty: Set δ=0
        - disable_recency_decay: Set γ=0
        - disable_failure_penalty: Set β=0
        - fifo_memory: Return insertion order (lower = earlier, for FIFO eviction)
        
        Args:
            current_step: Current step counter for recency calculation.
            params: Scoring hyperparameters (uses defaults if None).
            
        Returns:
            Retention score (higher = better, should be retained).
            If fifo_memory=True, returns negative insertion order (lower = earlier).
        """
        if params is None:
            params = ScoringParams()
        
        # FIFO mode: return negative creation timestamp (earlier = lower score = evicted first)
        if params.fifo_memory:
            return -self.created_at
        
        n_used = self.total_uses()
        
        # Success ratio term: α · (N_succ / (N_used + ε))
        success_term = params.alpha * (self.success_count / (n_used + params.epsilon))
        
        # Failure ratio term: β · (N_fail / (N_used + ε))
        # Disabled if disable_failure_penalty is True
        if params.disable_failure_penalty:
            failure_term = 0.0
        else:
            failure_term = params.beta * (self.failure_count / (n_used + params.epsilon))
        
        # Recency term: γ · exp(-λ · (t - t_last))
        # Disabled if disable_recency_decay is True
        if params.disable_recency_decay:
            recency_term = 0.0
        else:
            age = max(0, current_step - self.last_used_at) if current_step > 0 else 0
            recency_term = params.gamma * math.exp(-params.lambda_decay * age)
        
        # Vagueness penalty: δ · V(l_i)
        # Disabled if disable_vagueness_penalty is True
        if params.disable_vagueness_penalty:
            vagueness_term = 0.0
        else:
            vagueness_term = params.delta * self.vagueness_score
        
        # Final score
        return success_term - failure_term + recency_term - vagueness_term
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookEntry":
        """Create from dictionary, handling missing fields gracefully."""
        # Handle legacy entries that may not have all fields
        defaults = {
            "success_count": d.get("helpful_count", 0),
            "failure_count": d.get("harmful_count", 0),
            "token_count": 0,
            "vagueness_score": 0.0,
            "helpful_count": 0,
            "harmful_count": 0,
            "last_seen_step": d.get("last_used_at", 0),
            "is_generic": False,
        }
        
        # Merge defaults with provided dict
        for key, default_val in defaults.items():
            if key not in d:
                d[key] = default_val
        
        return cls(**d)


class Playbook:
    """ACE-style playbook storing domain-specific strategies.
    
    Supports two modes:
    - ace_full: Top-k entries per query (unbounded playbook size)
    - ace_working_memory: Token-budgeted entries (limited context window)
    
    The playbook implements strategic forgetting via:
    - Retention scoring (success rate, recency, vagueness)
    - Token-budget eviction (remove lowest-score entries when over budget)
    """
    
    def __init__(
        self,
        entries: Optional[List[PlaybookEntry]] = None,
        token_budget: Optional[int] = None,
        scoring_params: Optional[ScoringParams] = None,
    ):
        """
        Initialize playbook.
        
        Args:
            entries: Initial entries (optional).
            token_budget: Maximum total tokens for working memory mode (None = unlimited).
            scoring_params: Hyperparameters for retention scoring.
        """
        self.entries: List[PlaybookEntry] = entries or []
        self._next_id = 1
        self.token_budget = token_budget
        self.scoring_params = scoring_params or ScoringParams()
    
    @property
    def total_tokens(self) -> int:
        """Return total token count across all entries."""
        return sum(e.token_count for e in self.entries)
    
    def get_domain_tokens(self, domain: str) -> int:
        """Return total token count for a specific domain."""
        return sum(e.token_count for e in self.entries if e.domain == domain)
    
    @classmethod
    def load(cls, path: Path, token_budget: Optional[int] = None) -> "Playbook":
        """
        Load playbook from a JSONL file.
        
        Args:
            path: Path to JSONL file.
            token_budget: Token budget for working memory mode.
            
        Returns:
            Playbook instance.
        """
        entries = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry_dict = json.loads(line)
                            entries.append(PlaybookEntry.from_dict(entry_dict))
                        except (json.JSONDecodeError, TypeError) as e:
                            # Skip malformed entries
                            print(f"Warning: Skipping malformed playbook entry: {e}")
                            continue
        
        playbook = cls(entries, token_budget=token_budget)
        
        # Set next_id to max existing id + 1
        if entries:
            max_id = 0
            for e in entries:
                try:
                    entry_id = int(e.id)
                    max_id = max(max_id, entry_id)
                except (ValueError, TypeError):
                    pass
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
    
    def get_top_k(
        self,
        domain: str,
        k: int = 5,
        current_step: int = 0,
    ) -> List[PlaybookEntry]:
        """
        Get top-k entries for a domain, ranked by retention score.
        
        Args:
            domain: Domain name (e.g., "finance", "medical").
            k: Number of entries to return.
            current_step: Current step counter for recency calculation.
            
        Returns:
            List of top-k PlaybookEntry objects.
        """
        domain_entries = [e for e in self.entries if e.domain == domain]
        domain_entries.sort(
            key=lambda e: e.score(current_step, self.scoring_params),
            reverse=True
        )
        return domain_entries[:k]
    
    def get_top_entries_for_budget(
        self,
        domain: str,
        token_budget: int,
        current_step: int = 0,
    ) -> List[PlaybookEntry]:
        """
        Get top entries for a domain that fit within a token budget.
        
        Entries are sorted by retention score (descending) and greedily included
        until the estimated total token count <= token_budget.
        
        Args:
            domain: Domain name (e.g., "finance", "medical").
            token_budget: Maximum number of tokens allowed.
            current_step: Current step counter for recency calculation.
            
        Returns:
            List of PlaybookEntry objects that fit within the budget.
        """
        domain_entries = [e for e in self.entries if e.domain == domain]
        domain_entries.sort(
            key=lambda e: e.score(current_step, self.scoring_params),
            reverse=True
        )
        
        selected_entries = []
        total_tokens = 0
        
        for entry in domain_entries:
            if total_tokens + entry.token_count <= token_budget:
                selected_entries.append(entry)
                total_tokens += entry.token_count
            else:
                break
        
        return selected_entries
    
    def _find_duplicate(self, domain: str, text: str) -> Optional[PlaybookEntry]:
        """
        Check if a similar entry already exists.
        
        Args:
            domain: Domain to search in.
            text: Text to check for duplicates.
            
        Returns:
            Existing entry if duplicate found, None otherwise.
        """
        text_lower = text.lower().strip()
        for entry in self.entries:
            if entry.domain == domain:
                entry_text_lower = entry.text.lower().strip()
                # Check if one contains the other (simple similarity)
                if text_lower in entry_text_lower or entry_text_lower in text_lower:
                    return entry
        return None
    
    def _evict_lowest_score_entries(
        self,
        domain: str,
        tokens_needed: int,
        current_step: int,
    ) -> int:
        """
        Evict lowest-scoring entries to free up tokens.
        
        Args:
            domain: Domain to evict from.
            tokens_needed: Minimum tokens to free.
            current_step: Current step for scoring.
            
        Returns:
            Number of tokens freed.
        """
        # Get domain entries sorted by score (ascending = lowest first)
        domain_entries = [e for e in self.entries if e.domain == domain]
        domain_entries.sort(
            key=lambda e: e.score(current_step, self.scoring_params),
            reverse=False
        )
        
        tokens_freed = 0
        entries_to_remove = []
        
        for entry in domain_entries:
            if tokens_freed >= tokens_needed:
                break
            entries_to_remove.append(entry.id)
            tokens_freed += entry.token_count
        
        # Remove evicted entries
        self.entries = [e for e in self.entries if e.id not in entries_to_remove]
        
        return tokens_freed
    
    def add_entry(
        self,
        domain: str,
        text: str,
        step: int,
        enforce_budget: bool = True,
    ) -> Optional[PlaybookEntry]:
        """
        Add a new entry to the playbook, with deduplication and optional eviction.
        
        If token_budget is set and enforce_budget is True, this will evict
        lowest-scoring entries to make room for the new entry.
        
        NOTE: New entries are added WITHOUT any success/failure feedback.
        Feedback should only be recorded when the entry is actually USED
        in a subsequent prompt.
        
        Args:
            domain: Domain name.
            text: Strategy/lesson text.
            step: Current step/index.
            enforce_budget: Whether to evict entries if over token budget.
            
        Returns:
            The added or existing PlaybookEntry, or None if eviction failed.
        """
        # Check for duplicates
        existing = self._find_duplicate(domain, text)
        if existing is not None:
            existing.last_used_at = step
            existing.last_seen_step = step
            return existing
        
        # Create new entry (no feedback yet)
        entry_id = str(self._next_id)
        self._next_id += 1
        
        entry = PlaybookEntry(
            id=entry_id,
            domain=domain,
            text=text,
            success_count=0,
            failure_count=0,
            last_used_at=step,
            last_seen_step=step,
        )
        
        # Check token budget
        if self.token_budget is not None and enforce_budget:
            domain_tokens = self.get_domain_tokens(domain)
            if domain_tokens + entry.token_count > self.token_budget:
                # Need to evict some entries
                tokens_needed = (domain_tokens + entry.token_count) - self.token_budget
                tokens_freed = self._evict_lowest_score_entries(
                    domain, tokens_needed, step
                )
                
                if tokens_freed < tokens_needed:
                    # Couldn't free enough - still add but log warning
                    pass
        
        self.entries.append(entry)
        return entry
    
    def record_feedback(self, entry_id: str, helpful: bool) -> bool:
        """
        Record feedback for an entry (success or failure).
        
        This should ONLY be called when an entry was actually used in a prompt.
        
        Args:
            entry_id: ID of the entry.
            helpful: True if the answer was correct, False otherwise.
            
        Returns:
            True if entry was found and updated, False otherwise.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                if helpful:
                    entry.success_count += 1
                    entry.helpful_count += 1  # Legacy
                else:
                    entry.failure_count += 1
                    entry.harmful_count += 1  # Legacy
                return True
        
        return False
    
    def mark_entry_used(self, entry_id: str, step: int) -> bool:
        """
        Mark an entry as used at a specific step (for recency tracking).
        
        Args:
            entry_id: ID of the entry.
            step: Current step counter.
            
        Returns:
            True if entry was found and updated, False otherwise.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                entry.last_used_at = step
                entry.last_seen_step = step  # Legacy
                return True
        return False
    
    def prune(
        self,
        max_entries_per_domain: int = 32,
        current_step: int = 0,
    ) -> int:
        """
        Prune playbook to keep only top entries per domain.
        
        Args:
            max_entries_per_domain: Maximum entries to keep per domain.
            current_step: Current step for scoring.
            
        Returns:
            Number of entries removed.
        """
        original_count = len(self.entries)
        domains = set(e.domain for e in self.entries)
        pruned_entries = []
        
        for domain in domains:
            domain_entries = [e for e in self.entries if e.domain == domain]
            domain_entries.sort(
                key=lambda e: e.score(current_step, self.scoring_params),
                reverse=True
            )
            pruned_entries.extend(domain_entries[:max_entries_per_domain])
        
        self.entries = pruned_entries
        return original_count - len(self.entries)
    
    def get_stats(self, domain: Optional[str] = None) -> dict:
        """
        Get statistics about the playbook.
        
        Args:
            domain: Optional domain to filter by.
            
        Returns:
            Dictionary with playbook statistics.
        """
        entries = self.entries
        if domain:
            entries = [e for e in entries if e.domain == domain]
        
        if not entries:
            return {
                "num_entries": 0,
                "total_tokens": 0,
                "avg_score": 0.0,
                "avg_success_rate": 0.0,
            }
        
        total_uses = sum(e.total_uses() for e in entries)
        total_successes = sum(e.success_count for e in entries)
        
        return {
            "num_entries": len(entries),
            "total_tokens": sum(e.token_count for e in entries),
            "avg_score": sum(e.score() for e in entries) / len(entries),
            "avg_success_rate": total_successes / max(1, total_uses),
            "domains": list(set(e.domain for e in entries)),
        }
