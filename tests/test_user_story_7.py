"""
Test cases for User Story 7: Reset Memory / Start New Session
Issue #7: https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/7
"""

import unittest
from unittest.mock import patch
import streamlit as st
import app   # your Streamlit app file


class TestUserStory7Scenario1(unittest.TestCase):
    """
    Scenario 1:
    Given a user has been discussing a conservative investment strategy,
    when they click "Reset Memory", then the bot clears all previous financial
    information, risk preferences, and conversation history.
    """

    @patch("streamlit.session_state", {})
    def test_reset_memory_clears_all_state_fields(self):
        # Arrange — Put fake data to mimic an active session
        st.session_state["investment_recommendations"] = {"risk": "Conservative"}
        st.session_state["suggested_allocation"] = {"allocation": {"Bonds": 70}}
        st.session_state["goal_state"] = {"subtasks": ["task"]}
        st.session_state["portfolio_analysis_only"] = {"portfolio": {}}
        st.session_state["glossary"] = [("bond", "…")]
        st.session_state["explanation"] = "test explanation"

        # Act — call your reset function
        app.reset_all_memory()

        # Assert — verify everything is cleared
        self.assertNotIn("investment_recommendations", st.session_state)
        self.assertNotIn("suggested_allocation", st.session_state)
        self.assertNotIn("goal_state", st.session_state)
        self.assertNotIn("portfolio_analysis_only", st.session_state)
        self.assertNotIn("glossary", st.session_state)
        self.assertNotIn("explanation", st.session_state)


class TestUserStory7Scenario2(unittest.TestCase):
    """
    Scenario 2:
    Given a user wants to explore a new investment scenario,
    when they click "Start New Session", then the bot treats
    them as a new user with fresh profile.
    """

    @patch("streamlit.session_state", {})
    def test_start_new_session_resets_profile_defaults(self):
        # Arrange
        st.session_state["investment_recommendations"] = {"risk": "Aggressive"}
        st.session_state["age"] = 45
        st.session_state["risk_tolerance"] = "Aggressive"

        # Act
        app.reset_profile_and_memory()

        # Assert — profile defaults
        self.assertEqual(st.session_state.get("age"), 25)
        self.assertEqual(st.session_state.get("risk_tolerance"), "Moderate")
        self.assertEqual(st.session_state.get("savings"), 10000)
        self.assertFalse(st.session_state.get("has_401k"))
        self.assertFalse(st.session_state.get("has_stocks"))

        # Assert — all memory removed
        self.assertNotIn("investment_recommendations", st.session_state)
        self.assertNotIn("goal_state", st.session_state)
        self.assertNotIn("suggested_allocation", st.session_state)


class TestUserStory7Scenario3(unittest.TestCase):
    """
    Scenario 3:
    Given a user entered incorrect financial information,
    when they reset memory, then all recommendations and 
    analyses based on wrong data are cleared.
    """

    @patch("streamlit.session_state", {})
    def test_reset_memory_removes_incorrect_data(self):
        # Arrange — put wrong values in session
        st.session_state["age"] = 99
        st.session_state["savings"] = 1000000
        st.session_state["investment_recommendations"] = {"risk": "Conservative"}
        st.session_state["portfolio_analysis_only"] = {"analysis": "fake analysis"}

        # Act
        app.reset_all_memory()

        # Assert
        self.assertNotIn("investment_recommendations", st.session_state)
        self.assertNotIn("portfolio_analysis_only", st.session_state)
        # Age/savings should also be wiped if part of profile memory
        self.assertNotIn("age", st.session_state)
        self.assertNotIn("savings", st.session_state)


class TestUserStory7Structure(unittest.TestCase):
    """
    Structure Test:
    Verify required reset functions exist.
    """

    def test_functions_exist(self):
        self.assertTrue(hasattr(app, "reset_all_memory"))
        self.assertTrue(hasattr(app, "reset_profile_and_memory"))


if __name__ == "__main__":
    unittest.main()
