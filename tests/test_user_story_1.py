"""
Test cases for User Story 1: Financial Situation Input and Goal Decomposition
Issue #1: https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/1

As a beginner investor who wants to start building wealth,
I want the bot to accept my current financial situation and break down my investing goal 
into clear learning steps, so that I can systematically understand how to achieve my 
financial objectives.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Set environment variable for testing
os.environ['OPENAI_API_KEY'] = 'test-key-12345'


class TestUserStory1Scenario1(unittest.TestCase):
    """
    Scenario 1: Given a user enters "I want to start investing with $10,000 and learn about stocks",
    when the bot processes it, then it generates a learning path with 4-5 phases.
    """
    
    @patch('openai.OpenAI')
    def test_goal_decomposition_generates_4_to_5_phases(self, mock_openai_class):
        """Test that the system generates 4-5 learning phases for investment goal"""
        # Import after patching
        from app import decompose_goal
        import app
        
        # Arrange - setup mock client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        app.client = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """1. Understand investment basics and your current financial position
2. Learn about stocks and bonds suitable for moderate risk tolerance
3. Build a starter portfolio strategy with your $10,000
4. Understand risk management for your age group
5. Plan for long-term wealth building toward retirement"""
        
        mock_client.chat.completions.create.return_value = mock_response
        
        state = {
            "goal": "I want to start investing with $10,000 and learn about stocks",
            "subtasks": [],
            "completed_tasks": [],
            "user_info": {
                "age": 25,
                "risk_tolerance": "Moderate",
                "savings": 10000,
                "has_401k": False,
                "retirement_balance": 0,
                "has_stocks": False,
                "stock_value": 0,
                "experience": "beginner",
                "time": "3-5 hours"
            },
            "messages": []
        }
        
        # Act
        result = decompose_goal(state)
        
        # Assert
        self.assertIn("subtasks", result)
        self.assertGreaterEqual(len(result["subtasks"]), 4, 
                                "Should generate at least 4 learning phases")
        self.assertLessEqual(len(result["subtasks"]), 5, 
                           "Should generate at most 5 learning phases")
        
        # Verify phases cover key topics
        tasks_text = " ".join([task["task"].lower() for task in result["subtasks"]])
        self.assertIn("investment", tasks_text, "Should include investment basics")
        self.assertIn("stock", tasks_text, "Should include learning about stocks")
        self.assertIn("portfolio", tasks_text, "Should include portfolio strategy")
        self.assertIn("risk", tasks_text, "Should include risk management")


class TestUserStory1Scenario2(unittest.TestCase):
    """
    Scenario 2: Given the user provides additional context like 
    "I'm 25 years old, risk-tolerant, saving for retirement", when the goal is decomposed,
    then the bot tailors the learning tasks to match their age and timeline.
    """
    
    @patch('openai.OpenAI')
    def test_tailored_learning_tasks_for_age_and_timeline(self, mock_openai_class):
        """Test that learning tasks are tailored to user's age and retirement timeline"""
        # Import after patching
        from app import decompose_goal
        import app
        
        # Arrange
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        app.client = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """1. Understand investment basics and time horizon advantages at age 25
2. Learn about aggressive growth stocks suitable for your risk tolerance
3. Build a retirement-focused portfolio strategy with your $10,000
4. Understand long-term compounding for 40-year timeline
5. Plan tax-advantaged retirement accounts for young investors"""
        
        mock_client.chat.completions.create.return_value = mock_response
        
        state = {
            "goal": "I want to save for retirement and learn about long-term investing",
            "subtasks": [],
            "completed_tasks": [],
            "user_info": {
                "age": 25,
                "risk_tolerance": "Aggressive",
                "savings": 10000,
                "has_401k": False,
                "retirement_balance": 0,
                "has_stocks": False,
                "stock_value": 0,
                "experience": "beginner",
                "time": "5-10 hours"
            },
            "messages": []
        }
        
        # Act
        result = decompose_goal(state)
        
        # Assert
        # Verify API was called with user's age and risk tolerance
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        
        self.assertIn("Age: 25", prompt, "Prompt should include user's age")
        self.assertIn("Aggressive", prompt, "Prompt should include risk tolerance")
        self.assertIn("retirement", prompt.lower(), "Prompt should include retirement context")
        
        # Verify tasks are generated
        self.assertGreater(len(result["subtasks"]), 0, "Should generate subtasks")
        
        # Verify tasks mention age-appropriate or timeline-appropriate concepts
        tasks_text = " ".join([task["task"].lower() for task in result["subtasks"]])
        age_timeline_keywords = ["retirement", "long-term", "timeline", "age", "young", "compounding"]
        has_age_context = any(keyword in tasks_text for keyword in age_timeline_keywords)
        self.assertTrue(has_age_context, "Tasks should reflect age and timeline considerations")


class TestUserStory1Scenario3(unittest.TestCase):
    """
    Scenario 3: Given a user enters their current assets (savings, 401k, existing stocks),
    when the bot processes it, then it acknowledges their starting point and suggests 
    next steps based on their current situation.
    """
    
    @patch('openai.OpenAI')
    def test_acknowledges_current_assets_and_suggests_next_steps(self, mock_openai_class):
        """Test that the bot acknowledges user's existing assets and provides relevant next steps"""
        # Import after patching
        from app import decompose_goal
        import app
        
        # Arrange
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        app.client = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """1. Review and optimize your existing $5,000 stock portfolio
2. Understand how to leverage your $15,000 401(k) for retirement
3. Learn to diversify beyond your current stock holdings
4. Develop strategy for your additional $10,000 in savings
5. Create integrated wealth management plan across all assets"""
        
        mock_client.chat.completions.create.return_value = mock_response
        
        state = {
            "goal": "I want to optimize my current investments and learn how to grow my wealth",
            "subtasks": [],
            "completed_tasks": [],
            "user_info": {
                "age": 30,
                "risk_tolerance": "Moderate",
                "savings": 10000,
                "has_401k": True,
                "retirement_balance": 15000,
                "has_stocks": True,
                "stock_value": 5000,
                "experience": "intermediate",
                "time": "3-5 hours"
            },
            "messages": []
        }
        
        # Act
        result = decompose_goal(state)
        
        # Assert
        # Verify API was called with current asset information
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        
        self.assertIn("$10,000", prompt, "Prompt should include savings amount")
        self.assertIn("401(k): True", prompt, "Prompt should acknowledge 401(k) existence")
        self.assertIn("$15,000", prompt, "Prompt should include 401(k) balance")
        self.assertIn("Owns Stocks: True", prompt, "Prompt should acknowledge stock ownership")
        self.assertIn("$5,000", prompt, "Prompt should include stock portfolio value")
        
        # Verify tasks are generated
        self.assertGreater(len(result["subtasks"]), 0, "Should generate subtasks")
        
        # Verify subtasks acknowledge existing assets
        tasks_text = " ".join([task["task"].lower() for task in result["subtasks"]])
        asset_keywords = ["existing", "current", "portfolio", "401", "stocks", "savings"]
        acknowledges_assets = any(keyword in tasks_text for keyword in asset_keywords)
        self.assertTrue(acknowledges_assets, 
                       "Tasks should acknowledge user's existing assets")


class TestGoalStateStructure(unittest.TestCase):
    """Test that the GoalState properly captures all required financial information"""
    
    def test_goal_state_includes_financial_situation_fields(self):
        """Test that GoalState can store all financial situation fields"""
        from app import GoalState
        
        state: GoalState = {
            "goal": "Learn about investing",
            "subtasks": [],
            "completed_tasks": [],
            "user_info": {
                "age": 25,
                "risk_tolerance": "Moderate",
                "savings": 10000,
                "has_401k": True,
                "retirement_balance": 5000,
                "has_stocks": False,
                "stock_value": 0,
                "experience": "beginner",
                "time": "3-5 hours"
            },
            "messages": []
        }
        
        # Assert all fields are present
        self.assertEqual(state["user_info"]["age"], 25)
        self.assertEqual(state["user_info"]["risk_tolerance"], "Moderate")
        self.assertEqual(state["user_info"]["savings"], 10000)
        self.assertTrue(state["user_info"]["has_401k"])
        self.assertEqual(state["user_info"]["retirement_balance"], 5000)
        self.assertFalse(state["user_info"]["has_stocks"])
        self.assertEqual(state["user_info"]["stock_value"], 0)


if __name__ == '__main__':
    unittest.main()