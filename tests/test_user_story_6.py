"""
Test cases for User Story 6: Explanation of Recommendations
Issue #6: https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/6

As a user receiving investment guidance,
I want the bot to explain why it made specific recommendations
So that I can understand the reasoning and learn investment principles
"""

import unittest
from unittest.mock import Mock, patch
import os

# Fake API Key for mocking environment
os.environ["OPENAI_API_KEY"] = "test-key-123"


class TestUserStory6Scenario1(unittest.TestCase):
    """
    Scenario 1:
    Given the bot recommends "allocate 30% to bonds",
    when the user asks "Why 30% and not 20%?",
    then the bot explains the reasoning based on
    age, risk tolerance, and investment timeline.
    """

    @patch("openai.OpenAI")
    def test_reasoning_for_bond_allocation(self, mock_openai_class):
        from app import explain_recommendation
        import app

        # Arrange mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        app.client = mock_client

        sample_reasoning = (
            "30% bonds is recommended because at age 25 with moderate risk "
            "tolerance and a long investment timeline, bonds help stabilize "
            "volatility while still allowing growth."
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = sample_reasoning
        mock_client.chat.completions.create.return_value = mock_response

        user_profile = {"age": 25, "risk_tolerance": "Moderate", "timeline": "long-term"}

        # Act
        response = explain_recommendation(
            user_question="Why 30% and not 20%?",
            last_recommendation="Allocate 30% to bonds.",
            user_info=user_profile
        )

        # Assert
        self.assertIn("30%", response)
        self.assertIn("age", response.lower())
        self.assertIn("risk", response.lower())
        self.assertIn("timeline", response.lower())


class TestUserStory6Scenario2(unittest.TestCase):
    """
    Scenario 2:
    Given the bot suggests avoiding crypto,
    when the user asks "Why not cryptocurrency?",
    then the bot explains risk relative to conservative risk profile
    and short investment timeline.
    """

    @patch("openai.OpenAI")
    def test_reasoning_for_avoiding_crypto(self, mock_openai_class):
        from app import explain_recommendation
        import app

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        app.client = mock_client

        sample_reasoning = (
            "Cryptocurrency was avoided because your conservative risk tolerance "
            "and short-term investment horizon do not align with crypto's high volatility."
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = sample_reasoning
        mock_client.chat.completions.create.return_value = mock_response

        user_profile = {"risk_tolerance": "Conservative", "timeline": "short-term"}

        response = explain_recommendation(
            user_question="Why not cryptocurrency?",
            last_recommendation="Avoid cryptocurrency.",
            user_info=user_profile
        )

        self.assertIn("conservative", response.lower())
        self.assertIn("volatility", response.lower())
        self.assertIn("short", response.lower())


class TestUserStory6Scenario3(unittest.TestCase):
    """
    Scenario 3:
    Given the bot prioritizes certain tasks,
    when the user asks "Why learn about bonds before stocks?",
    then the bot explains the pedagogical reasoning.
    """

    @patch("openai.OpenAI")
    def test_reasoning_for_task_order(self, mock_openai_class):
        from app import explain_recommendation
        import app

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        app.client = mock_client

        sample_reasoning = (
            "Bonds are taught first because understanding fixed-income assets "
            "helps you grasp baseline stability before learning about the higher "
            "volatility of stocks."
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = sample_reasoning
        mock_client.chat.completions.create.return_value = mock_response

        response = explain_recommendation(
            user_question="Why learn about bonds before stocks?",
            last_recommendation="Learn bonds before stocks.",
            user_info={"experience": "beginner"}
        )

        self.assertIn("bonds", response.lower())
        self.assertIn("stocks", response.lower())
        self.assertIn("understanding", response.lower())
        self.assertIn("volatility", response.lower())


# --------- Structural Test ---------
class TestUserStory6Structure(unittest.TestCase):
    """Verify explain_recommendation exists and works with required params"""

    def test_function_signature_exists(self):
        from app import explain_recommendation
        sig = explain_recommendation.__code__.co_varnames
        self.assertIn("user_question", sig)
        self.assertIn("last_recommendation", sig)
        self.assertIn("user_info", sig)


if __name__ == "__main__":
    unittest.main()

