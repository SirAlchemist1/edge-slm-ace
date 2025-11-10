"""Tests for playbook functionality."""

import json
import tempfile
from pathlib import Path

from slm_ace.playbook import Playbook, PlaybookEntry


def test_playbook_add_and_save():
    """Test adding entries and saving playbook."""
    playbook = Playbook()
    
    # Add entries
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
        
        # Load back
        loaded = Playbook.load(temp_path)
        assert len(loaded.entries) == 3
        assert loaded.entries[0].text == "Calculate revenue before tax"
    finally:
        temp_path.unlink()


def test_playbook_prune():
    """Test pruning playbook to top entries."""
    playbook = Playbook()
    
    # Add multiple entries for finance domain
    for i in range(5):
        entry = playbook.add_entry("finance", f"Strategy {i}", step=i)
        # Make some entries better than others
        if i < 2:
            entry.helpful_count = 5
        else:
            entry.helpful_count = 1
    
    assert len(playbook.entries) == 5
    
    # Prune to top 2 per domain
    playbook.prune(max_entries_per_domain=2)
    
    # Should keep top 2 by score
    finance_entries = [e for e in playbook.entries if e.domain == "finance"]
    assert len(finance_entries) <= 2
    
    # Top entries should have higher helpful_count
    helpful_counts = [e.helpful_count for e in finance_entries]
    assert all(count >= 1 for count in helpful_counts)


def test_playbook_get_top_k():
    """Test getting top-k entries for a domain."""
    playbook = Playbook()
    
    # Add entries with different scores
    for i in range(5):
        entry = playbook.add_entry("finance", f"Strategy {i}", step=i)
        entry.helpful_count = 5 - i  # Decreasing helpfulness
    
    top_3 = playbook.get_top_k("finance", k=3)
    assert len(top_3) == 3
    
    # Should be sorted by score (descending)
    scores = [e.score() for e in top_3]
    assert scores == sorted(scores, reverse=True)


def test_playbook_record_feedback():
    """Test recording feedback for entries."""
    playbook = Playbook()
    
    entry = playbook.add_entry("finance", "Test strategy", step=1)
    assert entry.helpful_count == 0
    assert entry.harmful_count == 0
    
    playbook.record_feedback(entry.id, helpful=True)
    assert entry.helpful_count == 1
    assert entry.harmful_count == 0
    
    playbook.record_feedback(entry.id, helpful=False)
    assert entry.helpful_count == 1
    assert entry.harmful_count == 1

