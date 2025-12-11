"""Tests for playbook functionality."""

import json
import math
import tempfile
from pathlib import Path

from edge_slm_ace.memory.playbook import (
    Playbook,
    PlaybookEntry,
    ScoringParams,
    compute_vagueness_score,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_DELTA,
    DEFAULT_LAMBDA,
)


class TestVaguenessScore:
    """Tests for the vagueness scoring function."""
    
    def test_generic_phrase_is_vague(self):
        """Generic phrases should have high vagueness scores."""
        vague_texts = [
            "Think carefully about the problem",
            "Pay attention to details",
            "Remember to check your work",
            "Be thorough",
        ]
        for text in vague_texts:
            score = compute_vagueness_score(text)
            assert score > 0.3, f"'{text}' should be vague (score={score})"
    
    def test_specific_lesson_is_not_vague(self):
        """Specific lessons should have low vagueness scores."""
        specific_texts = [
            "For percentage calculations: divide by 100, then multiply by the base amount",
            "When calculating profit margin: (revenue - expenses) / revenue * 100",
            "If the question asks for net profit after tax, apply the tax rate to pre-tax profit",
        ]
        for text in specific_texts:
            score = compute_vagueness_score(text)
            assert score < 0.4, f"'{text}' should be specific (score={score})"
    
    def test_short_text_is_vague(self):
        """Very short text should be considered vague."""
        short_texts = ["Be good", "Try harder", "Think"]
        for text in short_texts:
            score = compute_vagueness_score(text)
            assert score >= 0.5, f"'{text}' is too short, should be vague (score={score})"


class TestPlaybookEntry:
    """Tests for PlaybookEntry scoring."""
    
    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = PlaybookEntry(
            id="1",
            domain="finance",
            text="Calculate profit by subtracting expenses from revenue",
        )
        assert entry.id == "1"
        assert entry.domain == "finance"
        assert entry.success_count == 0
        assert entry.failure_count == 0
        assert entry.token_count > 0  # Auto-computed
        assert 0 <= entry.vagueness_score <= 1  # Auto-computed
    
    def test_score_formula_success_term(self):
        """Test that success increases score."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson with some content")
        
        # Initial score (no uses)
        initial_score = entry.score(current_step=10)
        
        # Add successes
        entry.success_count = 5
        entry.failure_count = 0
        entry.last_used_at = 10
        
        score_with_success = entry.score(current_step=10)
        
        # Score should increase with successes
        assert score_with_success > initial_score
    
    def test_score_formula_failure_term(self):
        """Test that failures decrease score."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson with some content")
        entry.last_used_at = 10
        
        # Add only failures
        entry.success_count = 0
        entry.failure_count = 5
        
        score_with_failures = entry.score(current_step=10)
        
        # Entry with only failures should have negative or low score
        assert score_with_failures < 0.5
    
    def test_score_formula_recency_decay(self):
        """Test that older entries have lower recency bonus."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson with some content")
        entry.success_count = 3
        entry.failure_count = 1
        entry.last_used_at = 5
        
        # Score at step 5 (just used)
        score_fresh = entry.score(current_step=5)
        
        # Score at step 100 (old)
        score_old = entry.score(current_step=100)
        
        # Fresh entry should have higher score due to recency
        assert score_fresh > score_old
    
    def test_score_formula_vagueness_penalty(self):
        """Test that vague entries have lower scores."""
        specific_entry = PlaybookEntry(
            id="1",
            domain="test",
            text="For percentage: divide by 100, multiply by base amount = result",
        )
        
        vague_entry = PlaybookEntry(
            id="2",
            domain="test",
            text="Think carefully",
        )
        
        # Both at same step, no usage history
        assert specific_entry.score() > vague_entry.score()
    
    def test_score_with_custom_params(self):
        """Test scoring with custom hyperparameters."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson here")
        entry.success_count = 5
        entry.failure_count = 2
        entry.last_used_at = 10
        
        # Default params
        default_score = entry.score(current_step=10)
        
        # High alpha (emphasize success)
        high_alpha = ScoringParams(alpha=2.0, beta=0.1)
        high_alpha_score = entry.score(current_step=10, params=high_alpha)
        
        assert high_alpha_score > default_score


class TestPlaybook:
    """Tests for Playbook class."""
    
    def test_add_and_save(self):
        """Test adding entries and saving playbook."""
        playbook = Playbook()
        
        entry1 = playbook.add_entry("finance", "Calculate revenue before tax", step=1)
        entry2 = playbook.add_entry("finance", "Subtract expenses from revenue", step=2)
        entry3 = playbook.add_entry("medical", "Check blood pressure first", step=1)
        
        assert len(playbook.entries) == 3
        assert entry1.domain == "finance"
        assert entry2.domain == "finance"
        assert entry3.domain == "medical"
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            playbook.save(temp_path)
            assert temp_path.exists()
            
            loaded = Playbook.load(temp_path)
            assert len(loaded.entries) == 3
            assert loaded.entries[0].text == "Calculate revenue before tax"
        finally:
            temp_path.unlink()
    
    def test_deduplication(self):
        """Test that duplicate entries are not added."""
        playbook = Playbook()
        
        entry1 = playbook.add_entry("finance", "Calculate revenue", step=1)
        entry2 = playbook.add_entry("finance", "Calculate revenue", step=2)  # Duplicate
        
        assert len(playbook.entries) == 1
        assert entry1.id == entry2.id  # Should return same entry
    
    def test_get_top_k(self):
        """Test getting top-k entries for a domain."""
        playbook = Playbook()
        
        # Add entries with different success rates
        for i in range(5):
            entry = playbook.add_entry("finance", f"Strategy number {i} with details", step=i)
            entry.success_count = 5 - i  # Decreasing success
            entry.failure_count = i
            entry.last_used_at = i
        
        top_3 = playbook.get_top_k("finance", k=3, current_step=10)
        assert len(top_3) == 3
        
        # Should be sorted by score (descending)
        scores = [e.score(current_step=10) for e in top_3]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_top_entries_for_budget(self):
        """Test token-budgeted entry retrieval."""
        playbook = Playbook()
        
        # Add entries of different sizes
        playbook.add_entry("finance", "Short", step=1)  # ~2 tokens
        playbook.add_entry("finance", "Medium length lesson with some details", step=2)  # ~9 tokens
        playbook.add_entry("finance", "A very long and detailed lesson with many words and specific instructions about financial calculations", step=3)  # ~20 tokens
        
        # Get entries within small budget
        entries = playbook.get_top_entries_for_budget("finance", token_budget=15, current_step=5)
        
        # Should not exceed budget
        total_tokens = sum(e.token_count for e in entries)
        assert total_tokens <= 15
    
    def test_record_feedback_only_for_used_entries(self):
        """Test that feedback is only recorded for entries that were used."""
        playbook = Playbook()
        
        entry = playbook.add_entry("finance", "Test strategy with content", step=1)
        
        # New entry should have no feedback
        assert entry.success_count == 0
        assert entry.failure_count == 0
        
        # Record feedback (simulating actual use)
        playbook.record_feedback(entry.id, helpful=True)
        assert entry.success_count == 1
        assert entry.failure_count == 0
        
        playbook.record_feedback(entry.id, helpful=False)
        assert entry.success_count == 1
        assert entry.failure_count == 1
    
    def test_token_budget_eviction(self):
        """Test that low-score entries are evicted when over budget."""
        # Create playbook with small token budget
        playbook = Playbook(token_budget=50)
        
        # Add entries until we hit the budget
        entries = []
        for i in range(5):
            entry = playbook.add_entry(
                "finance",
                f"Lesson {i}: This is a detailed lesson with specific content about topic {i}",
                step=i,
                enforce_budget=True,
            )
            entries.append(entry)
            
            # Give some entries better scores
            if i < 2:
                entry.success_count = 5
            else:
                entry.failure_count = 3
        
        # Add one more entry that should trigger eviction
        playbook.add_entry(
            "finance",
            "New lesson: Another detailed lesson that should trigger eviction of low-score entries",
            step=10,
            enforce_budget=True,
        )
        
        # Total tokens should be within budget (or close)
        domain_tokens = playbook.get_domain_tokens("finance")
        assert domain_tokens <= playbook.token_budget + 50  # Some tolerance
    
    def test_prune_keeps_top_entries(self):
        """Test that pruning keeps highest-scoring entries."""
        playbook = Playbook()
        
        # Add entries with different scores
        for i in range(10):
            entry = playbook.add_entry("finance", f"Strategy {i} with some content", step=i)
            entry.success_count = 10 - i  # Higher index = lower score
            entry.last_used_at = i
        
        assert len(playbook.entries) == 10
        
        # Prune to top 3
        removed = playbook.prune(max_entries_per_domain=3, current_step=15)
        
        assert len(playbook.entries) == 3
        assert removed == 7
        
        # Remaining entries should have highest success counts
        success_counts = [e.success_count for e in playbook.entries]
        assert all(count >= 8 for count in success_counts)
    
    def test_get_stats(self):
        """Test playbook statistics."""
        playbook = Playbook()
        
        # Add some entries
        for i in range(3):
            entry = playbook.add_entry("finance", f"Lesson {i} about finance topics", step=i)
            entry.success_count = 2
            entry.failure_count = 1
        
        stats = playbook.get_stats("finance")
        
        assert stats["num_entries"] == 3
        assert stats["total_tokens"] > 0
        assert stats["avg_success_rate"] > 0
        assert "finance" in stats["domains"]


class TestPlaybookLegacyCompatibility:
    """Tests for backward compatibility with legacy playbook format."""
    
    def test_load_legacy_entry(self):
        """Test loading entries from legacy format."""
        legacy_dict = {
            "id": "1",
            "domain": "finance",
            "text": "Test lesson",
            "helpful_count": 5,
            "harmful_count": 2,
            "created_at": 1234567890.0,
            "last_seen_step": 10,
            "is_generic": False,
        }
        
        entry = PlaybookEntry.from_dict(legacy_dict)
        
        assert entry.id == "1"
        assert entry.success_count == 5  # Migrated from helpful_count
        assert entry.failure_count == 2  # Migrated from harmful_count
    
    def test_save_includes_legacy_fields(self):
        """Test that saved entries include legacy fields."""
        entry = PlaybookEntry(
            id="1",
            domain="finance",
            text="Test lesson",
        )
        entry.success_count = 3
        entry.failure_count = 1
        
        entry_dict = entry.to_dict()
        
        # Should include both new and legacy fields
        assert "success_count" in entry_dict
        assert "failure_count" in entry_dict
        assert "helpful_count" in entry_dict
        assert "harmful_count" in entry_dict


# Legacy test names for backward compatibility
def test_playbook_add_and_save():
    """Test adding entries and saving playbook."""
    TestPlaybook().test_add_and_save()


def test_playbook_prune():
    """Test pruning playbook to top entries."""
    TestPlaybook().test_prune_keeps_top_entries()


def test_playbook_get_top_k():
    """Test getting top-k entries for a domain."""
    TestPlaybook().test_get_top_k()


def test_playbook_record_feedback():
    """Test recording feedback for entries."""
    TestPlaybook().test_record_feedback_only_for_used_entries()


class TestAblationFlags:
    """Tests for retention scoring ablation flags."""
    
    def test_disable_vagueness_penalty(self):
        """When disable_vagueness_penalty=True, vagueness term should be zero."""
        entry = PlaybookEntry(
            id="1",
            domain="test",
            text="Think carefully",  # Vague text
            success_count=1,
            failure_count=0,
            last_used_at=0,
        )
        
        # With vagueness penalty enabled (default)
        params_default = ScoringParams()
        score_with_penalty = entry.score(current_step=1, params=params_default)
        
        # With vagueness penalty disabled
        params_no_vagueness = ScoringParams(disable_vagueness_penalty=True)
        score_without_penalty = entry.score(current_step=1, params=params_no_vagueness)
        
        # Score without penalty should be higher (less negative)
        assert score_without_penalty > score_with_penalty, \
            "Disabling vagueness penalty should increase score"
    
    def test_disable_recency_decay(self):
        """When disable_recency_decay=True, recency term should be zero."""
        entry = PlaybookEntry(
            id="1",
            domain="test",
            text="Test lesson",
            success_count=1,
            failure_count=0,
            last_used_at=0,  # Used at step 0
        )
        
        # With recency enabled (default), recent entries get bonus
        params_default = ScoringParams()
        score_recent = entry.score(current_step=1, params=params_default)  # Recent
        
        # With recency disabled
        params_no_recency = ScoringParams(disable_recency_decay=True)
        score_no_recency = entry.score(current_step=1, params=params_no_recency)
        
        # Score without recency should be lower (no recency bonus)
        assert score_no_recency < score_recent, \
            "Disabling recency should remove recency bonus"
    
    def test_disable_failure_penalty(self):
        """When disable_failure_penalty=True, failure term should be zero."""
        entry = PlaybookEntry(
            id="1",
            domain="test",
            text="Test lesson",
            success_count=1,
            failure_count=2,  # Has failures
            last_used_at=0,
        )
        
        # With failure penalty enabled (default)
        params_default = ScoringParams()
        score_with_penalty = entry.score(current_step=1, params=params_default)
        
        # With failure penalty disabled
        params_no_failure = ScoringParams(disable_failure_penalty=True)
        score_without_penalty = entry.score(current_step=1, params=params_no_failure)
        
        # Score without penalty should be higher (no failure penalty)
        assert score_without_penalty > score_with_penalty, \
            "Disabling failure penalty should increase score"
    
    def test_fifo_memory(self):
        """When fifo_memory=True, scoring should return insertion order."""
        # Create entries at different times
        import time
        entry1 = PlaybookEntry(
            id="1",
            domain="test",
            text="First entry",
            created_at=time.time() - 10,  # Older
        )
        entry2 = PlaybookEntry(
            id="2",
            domain="test",
            text="Second entry",
            created_at=time.time(),  # Newer
        )
        
        params_fifo = ScoringParams(fifo_memory=True)
        
        score1 = entry1.score(current_step=1, params=params_fifo)
        score2 = entry2.score(current_step=1, params=params_fifo)
        
        # Older entry should have lower score (evicted first)
        assert score1 < score2, \
            "In FIFO mode, older entries should have lower scores"
        
        # Scores should be negative (based on negative created_at)
        assert score1 < 0 and score2 < 0, \
            "FIFO scores should be negative (negative created_at)"
