"""
Test cases for User Story 10: Account Information and Portfolio Snapshot
Reference: GitHub Issue #10

As a returning user who has previously described my financial situation
I want to save my portfolio information and reload it in future sessions
So that I don't have to re-enter my financial details every time I use the bot

Run with: python tests/test_user_story_10.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import shutil
from datetime import datetime
import streamlit as st
from app import (
    save_portfolio_snapshot, 
    load_all_saved_portfolios, 
    delete_portfolio_snapshot,
    apply_pending_load,
    serialize_snapshot, 
    deserialize_snapshot,
    PORTFOLIOS_DIR
)


def setup_test_environment():
    """Setup test environment and clean up any existing test files"""
    # Clear session state
    st.session_state.clear()
    
    # Clean up test portfolios directory
    if os.path.exists(PORTFOLIOS_DIR):
        shutil.rmtree(PORTFOLIOS_DIR)
    os.makedirs(PORTFOLIOS_DIR)


def setup_mock_session_state():
    """Setup mock session state for testing"""
    st.session_state.clear()
    
    # Mock profile data
    st.session_state["age"] = 30
    st.session_state["risk_tolerance"] = "Aggressive"
    st.session_state["savings"] = 25000
    st.session_state["has_401k"] = True
    st.session_state["retirement_balance"] = 50000
    st.session_state["has_stocks"] = True
    st.session_state["stock_value"] = 15000
    st.session_state["experience"] = "Intermediate"
    st.session_state["time_available"] = "5-10 hours"
    
    # Mock portfolio allocation
    st.session_state["stocks"] = 60.0
    st.session_state["bonds"] = 30.0
    st.session_state["cash"] = 10.0
    st.session_state["real_estate"] = 0.0
    st.session_state["crypto"] = 0.0
    st.session_state["other"] = 0.0
    
    # Mock goal state
    st.session_state["goal_state"] = {
        "goal": "Learn about retirement investing",
        "subtasks": [
            {"id": 1, "task": "Understand 401k basics", "completed": True},
            {"id": 2, "task": "Learn about index funds", "completed": False}
        ],
        "completed_tasks": [],
        "user_info": {},
        "messages": [],
        "portfolio": {},
        "portfolio_analysis": ""
    }
    
    # Mock glossary
    st.session_state["glossary"] = [
        ("What is a 401k?", "A retirement account..."),
        ("What is an index fund?", "A type of mutual fund...")
    ]


# Scenario 1: Save Portfolio
def test_save_portfolio_snapshot():
    """
    Scenario 1: Given a user has input their current holdings and completed several learning tasks,
    when they click "Save Portfolio", then the bot stores their financial snapshot
    and learning progress state.
    """
    # Arrange
    setup_test_environment()
    setup_mock_session_state()
    portfolio_name = "My Retirement Portfolio"
    
    # Act
    snapshot = save_portfolio_snapshot(portfolio_name)
    
    # Assert - Check snapshot structure
    assert snapshot is not None, "Snapshot should be created"
    assert "name" in snapshot, "Snapshot should have name"
    assert "timestamp" in snapshot, "Snapshot should have timestamp"
    assert "profile" in snapshot, "Snapshot should have profile"
    assert "portfolio" in snapshot, "Snapshot should have portfolio"
    assert "goal_state" in snapshot, "Snapshot should have goal_state"
    assert "glossary" in snapshot, "Snapshot should have glossary"
    
    # Assert - Check name
    assert snapshot["name"] == portfolio_name, "Name should match"
    
    # Assert - Check profile data
    profile = snapshot["profile"]
    assert profile["age"] == 30, "Age should be saved"
    assert profile["risk_tolerance"] == "Aggressive", "Risk tolerance should be saved"
    assert profile["savings"] == 25000, "Savings should be saved"
    assert profile["has_401k"] == True, "401k status should be saved"
    assert profile["retirement_balance"] == 50000, "Retirement balance should be saved"
    assert profile["has_stocks"] == True, "Stock ownership should be saved"
    assert profile["stock_value"] == 15000, "Stock value should be saved"
    assert profile["experience"] == "Intermediate", "Experience should be saved"
    assert profile["time_available"] == "5-10 hours", "Time available should be saved"
    
    # Assert - Check portfolio allocation
    portfolio = snapshot["portfolio"]
    assert portfolio["stocks"] == 60.0, "Stock allocation should be saved"
    assert portfolio["bonds"] == 30.0, "Bond allocation should be saved"
    assert portfolio["cash"] == 10.0, "Cash allocation should be saved"
    
    # Assert - Check goal state
    goal_state = snapshot["goal_state"]
    assert goal_state is not None, "Goal state should be saved"
    assert goal_state["goal"] == "Learn about retirement investing", "Goal should be saved"
    assert len(goal_state["subtasks"]) == 2, "Subtasks should be saved"
    assert goal_state["subtasks"][0]["completed"] == True, "Task completion should be saved"
    
    # Assert - Check glossary
    glossary = snapshot["glossary"]
    assert len(glossary) == 2, "Glossary should be saved"
    assert glossary[0][0] == "What is a 401k?", "Glossary terms should be saved"
    
    # Assert - Check file was created
    saved_files = os.listdir(PORTFOLIOS_DIR)
    assert len(saved_files) == 1, "Should create one file"
    assert saved_files[0].endswith('.json'), "File should be JSON"
    
    print(f"\n✅ Scenario 1 PASSED")
    print(f"Portfolio Name: {portfolio_name}")
    print(f"Snapshot created with {len(profile)} profile fields")
    print(f"Goal: {goal_state['goal']}")
    print(f"Completed tasks: {sum(1 for t in goal_state['subtasks'] if t['completed'])}/{len(goal_state['subtasks'])}")
    print(f"File saved: {saved_files[0]}")


# Scenario 2: Load Previous Session
def test_load_portfolio_snapshot():
    """
    Scenario 2: Given a user returns to the application after closing it,
    when they click "Load Previous Session", then the bot restores their
    portfolio details, risk tolerance settings, and conversation context.
    """
    # Arrange
    setup_test_environment()
    setup_mock_session_state()
    
    # Save snapshot
    save_portfolio_snapshot("Test Portfolio")
    
    # Clear session state (simulate closing app)
    st.session_state.clear()
    
    # Verify state is cleared
    assert st.session_state.get("age") is None, "Age should be cleared"
    assert st.session_state.get("goal_state") is None, "Goal state should be cleared"
    
    # Act - Load from disk
    saved_portfolios = load_all_saved_portfolios()
    assert len(saved_portfolios) == 1, "Should load one portfolio"
    
    # Simulate the load process
    st.session_state["_pending_load"] = saved_portfolios[0]
    apply_pending_load()
    
    # Assert - Check profile restored
    assert st.session_state["age"] == 30, "Age should be restored"
    assert st.session_state["risk_tolerance"] == "Aggressive", "Risk tolerance should be restored"
    assert st.session_state["savings"] == 25000, "Savings should be restored"
    assert st.session_state["has_401k"] == True, "401k status should be restored"
    assert st.session_state["retirement_balance"] == 50000, "Retirement balance should be restored"
    assert st.session_state["has_stocks"] == True, "Stock ownership should be restored"
    assert st.session_state["stock_value"] == 15000, "Stock value should be restored"
    assert st.session_state["experience"] == "Intermediate", "Experience should be restored"
    assert st.session_state["time_available"] == "5-10 hours", "Time available should be restored"
    
    # Assert - Check portfolio allocation restored
    assert st.session_state["stocks"] == 60.0, "Stock allocation should be restored"
    assert st.session_state["bonds"] == 30.0, "Bond allocation should be restored"
    assert st.session_state["cash"] == 10.0, "Cash allocation should be restored"
    
    # Assert - Check goal state restored
    goal_state = st.session_state["goal_state"]
    assert goal_state is not None, "Goal state should be restored"
    assert goal_state["goal"] == "Learn about retirement investing", "Goal should be restored"
    assert len(goal_state["subtasks"]) == 2, "Subtasks should be restored"
    assert goal_state["subtasks"][0]["completed"] == True, "Task completion should be restored"
    
    # Assert - Check glossary restored
    glossary = st.session_state["glossary"]
    assert len(glossary) == 2, "Glossary should be restored"
    
    # Assert - Check pending load was cleared
    assert "_pending_load" not in st.session_state, "Pending load should be cleared"
    
    print(f"\n✅ Scenario 2 PASSED")
    print(f"Session restored successfully from disk")
    print(f"Profile: {st.session_state['age']} years old, {st.session_state['risk_tolerance']} risk")
    print(f"Portfolio: {st.session_state['stocks']}% stocks, {st.session_state['bonds']}% bonds")
    print(f"Progress: {sum(1 for t in goal_state['subtasks'] if t['completed'])}/{len(goal_state['subtasks'])} tasks completed")


# Scenario 3: Multiple Saved Portfolios
def test_multiple_saved_portfolios():
    """
    Scenario 3: Given a user has multiple saved portfolios,
    when they view their saved sessions, then they can select which
    portfolio snapshot to load or delete old sessions they no longer need.
    """
    # Arrange
    setup_test_environment()
    
    # Create first portfolio (Conservative)
    st.session_state["age"] = 55
    st.session_state["risk_tolerance"] = "Conservative"
    st.session_state["savings"] = 100000
    st.session_state["stocks"] = 30.0
    st.session_state["bonds"] = 60.0
    st.session_state["cash"] = 10.0
    st.session_state["real_estate"] = 0.0
    st.session_state["crypto"] = 0.0
    st.session_state["other"] = 0.0
    st.session_state["goal_state"] = {"goal": "Preserve wealth for retirement", "subtasks": []}
    st.session_state["glossary"] = []
    st.session_state["has_401k"] = False
    st.session_state["retirement_balance"] = 0
    st.session_state["has_stocks"] = False
    st.session_state["stock_value"] = 0
    st.session_state["experience"] = "Advanced"
    st.session_state["time_available"] = "1-2 hours"
    
    save_portfolio_snapshot("Conservative Retirement")
    
    # Create second portfolio (Aggressive)
    st.session_state["age"] = 25
    st.session_state["risk_tolerance"] = "Aggressive"
    st.session_state["savings"] = 15000
    st.session_state["stocks"] = 80.0
    st.session_state["bonds"] = 15.0
    st.session_state["cash"] = 5.0
    st.session_state["goal_state"] = {"goal": "Maximize growth", "subtasks": []}
    st.session_state["experience"] = "Beginner"
    
    save_portfolio_snapshot("Aggressive Growth")
    
    # Create third portfolio (Moderate)
    st.session_state["age"] = 35
    st.session_state["risk_tolerance"] = "Moderate"
    st.session_state["savings"] = 50000
    st.session_state["stocks"] = 60.0
    st.session_state["bonds"] = 35.0
    st.session_state["cash"] = 5.0
    st.session_state["goal_state"] = {"goal": "Balanced approach", "subtasks": []}
    st.session_state["experience"] = "Intermediate"
    
    save_portfolio_snapshot("Balanced Portfolio")
    
    # Act - Load all portfolios from disk
    saved_portfolios = load_all_saved_portfolios()
    
    # Assert - Check all portfolios saved
    assert len(saved_portfolios) == 3, "Should have 3 saved portfolios"
    
    # Assert - Check portfolios are sorted by timestamp (newest first)
    assert saved_portfolios[0]["name"] == "Balanced Portfolio", "Newest should be first"
    
    # Act - Load first portfolio (newest)
    st.session_state.clear()
    st.session_state["_pending_load"] = saved_portfolios[0]
    apply_pending_load()
    
    # Assert - Check first portfolio loaded correctly
    assert st.session_state["age"] == 35, "Should load balanced portfolio age"
    assert st.session_state["risk_tolerance"] == "Moderate", "Should load moderate risk"
    assert st.session_state["stocks"] == 60.0, "Should load moderate allocation"
    
    # Act - Load last portfolio (oldest - conservative)
    st.session_state.clear()
    st.session_state["_pending_load"] = saved_portfolios[2]
    apply_pending_load()
    
    # Assert - Check last portfolio loaded correctly
    assert st.session_state["age"] == 55, "Should load conservative portfolio age"
    assert st.session_state["risk_tolerance"] == "Conservative", "Should load conservative risk"
    assert st.session_state["stocks"] == 30.0, "Should load conservative allocation"
    
    # Act - Delete middle portfolio (Aggressive Growth)
    filename_to_delete = saved_portfolios[1]["filename"]
    delete_portfolio_snapshot(filename_to_delete)
    
    # Reload portfolios from disk
    saved_portfolios = load_all_saved_portfolios()
    
    # Assert - Check deletion
    assert len(saved_portfolios) == 2, "Should have 2 portfolios after deletion"
    portfolio_names = [p["name"] for p in saved_portfolios]
    assert "Aggressive Growth" not in portfolio_names, "Deleted portfolio should not be in list"
    assert "Conservative Retirement" in portfolio_names, "First portfolio should remain"
    assert "Balanced Portfolio" in portfolio_names, "Third portfolio should remain"
    
    # Assert - Check file was actually deleted
    assert not os.path.exists(os.path.join(PORTFOLIOS_DIR, filename_to_delete)), \
        "File should be deleted from disk"
    
    print(f"\n✅ Scenario 3 PASSED")
    print(f"Created 3 portfolios")
    print(f"Successfully loaded different portfolios")
    print(f"Successfully deleted portfolio from disk")
    print(f"Remaining portfolios: {portfolio_names}")


# Additional Test: Persistence Across Sessions
def test_persistence_across_sessions():
    """Test that portfolios persist even after clearing session state"""
    # Arrange
    setup_test_environment()
    setup_mock_session_state()
    
    # Save portfolio
    save_portfolio_snapshot("Persistent Portfolio")
    
    # Completely clear session state (simulate app restart)
    st.session_state.clear()
    
    # Act - Load from disk (simulating new session)
    saved_portfolios = load_all_saved_portfolios()
    
    # Assert - Portfolio should still exist
    assert len(saved_portfolios) == 1, "Portfolio should persist"
    assert saved_portfolios[0]["name"] == "Persistent Portfolio", "Name should match"
    assert saved_portfolios[0]["profile"]["age"] == 30, "Data should be intact"
    
    print(f"\n✅ Persistence test PASSED")
    print("Portfolio survived session clear")


# Additional Test: Serialization/Deserialization
def test_snapshot_serialization():
    """Test that snapshots can be serialized and deserialized for export/import"""
    # Arrange
    setup_test_environment()
    setup_mock_session_state()
    save_portfolio_snapshot("Test Export")
    
    saved_portfolios = load_all_saved_portfolios()
    original_snapshot = saved_portfolios[0]
    
    # Act - Serialize
    serialized = serialize_snapshot(original_snapshot)
    
    # Assert - Check serialization
    assert serialized is not None, "Serialization should produce output"
    assert len(serialized) > 0, "Serialized data should not be empty"
    assert isinstance(serialized, str), "Serialized data should be string"
    
    # Act - Deserialize
    deserialized_snapshot = deserialize_snapshot(serialized)
    
    # Assert - Check deserialization
    assert deserialized_snapshot is not None, "Deserialization should work"
    assert deserialized_snapshot["name"] == "Test Export", "Name should match"
    assert deserialized_snapshot["profile"]["age"] == 30, "Profile data should match"
    assert deserialized_snapshot["portfolio"]["stocks"] == 60.0, "Portfolio data should match"
    
    # Assert - Filename should not be in serialized version
    assert "filename" not in deserialized_snapshot, "Filename should be excluded from export"
    
    print(f"\n✅ Serialization test PASSED")
    print(f"Serialized length: {len(serialized)} characters")
    print(f"Successfully round-tripped snapshot data")


# Additional Test: Empty State Handling
def test_save_empty_state():
    """Test saving when session state is mostly empty"""
    # Arrange
    setup_test_environment()
    st.session_state.clear()
    
    # Act
    snapshot = save_portfolio_snapshot("Empty Portfolio")
    
    # Assert
    assert snapshot is not None, "Should create snapshot even with empty state"
    assert snapshot["profile"]["age"] == 25, "Should use default age"
    assert snapshot["profile"]["risk_tolerance"] == "Moderate", "Should use default risk"
    assert snapshot["portfolio"]["stocks"] == 0.0, "Should use default allocation"
    
    # Load and verify
    saved_portfolios = load_all_saved_portfolios()
    assert len(saved_portfolios) == 1, "Should save empty state"
    
    print(f"\n✅ Empty state test PASSED")
    print("Empty state handled with defaults and persisted")


# Cleanup after all tests
def cleanup_test_files():
    """Clean up test directory after tests"""
    if os.path.exists(PORTFOLIOS_DIR):
        shutil.rmtree(PORTFOLIOS_DIR)
        os.makedirs(PORTFOLIOS_DIR)


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("Testing Portfolio Save/Load User Story")
    print("="*60)
    
    try:
        test_save_portfolio_snapshot()
        test_load_portfolio_snapshot()
        test_multiple_saved_portfolios()
        test_persistence_across_sessions()
        test_snapshot_serialization()
        test_save_empty_state()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
    finally:
        # Clean up test files
        cleanup_test_files()
        print("\n🧹 Test files cleaned up")