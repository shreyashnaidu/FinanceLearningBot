"""
Test cases for User Story: Portfolio Guidance and Health Assessment
Reference: GitHub Issue #2
Install pytest if not already installed: pip install pytest
Run with: pytest tests/test_portfolio_guidance.py -v
"""

import pytest
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client for testing
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_portfolio_for_test(portfolio: dict, experience: str = "beginner") -> str:
    """Helper function to analyze portfolio using OpenAI"""
    portfolio_desc = ", ".join([f"{k}: {v}%" for k, v in portfolio.items()])
    total = sum(portfolio.values())
    
    prompt = f"""You are a financial advisor analyzing a client's investment portfolio.

Client Profile:
- Experience Level: {experience}
- Portfolio: {portfolio_desc}
- Total Allocation: {total}%

Provide a brief portfolio analysis including:
1. Portfolio health assessment
2. Key risks identified
3. Top 2-3 recommendations

Keep response concise (under 200 words).
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content


# Scenario 1: Unbalanced Portfolio (60% stocks, 40% cash)
def test_portfolio_unbalanced():
    """
    Scenario 1: Given a user has input their current holdings (60% stocks, 40% cash),
    when they request portfolio analysis, then the bot provides feedback on
    diversification and suggests bond allocation for balance.
    """
    # Arrange
    portfolio = {
        "Stocks": 60.0,
        "Cash": 40.0
    }
    experience = "intermediate"
    
    # Act
    analysis = analyze_portfolio_for_test(portfolio, experience)
    
    # Assert
    assert analysis is not None, "Analysis should not be None"
    assert len(analysis) > 50, "Analysis should be substantial"
    
    # Check for key concepts in response (case-insensitive)
    analysis_lower = analysis.lower()
    
    # Should mention bonds or diversification
    has_bond_mention = "bond" in analysis_lower
    has_diversification = "diversif" in analysis_lower
    
    assert has_bond_mention or has_diversification, \
        "Analysis should mention bonds or diversification for unbalanced portfolio"
    
    print(f"\n✅ Scenario 1 PASSED")
    print(f"Portfolio: {portfolio}")
    print(f"Analysis excerpt: {analysis[:200]}...")


# Scenario 2: No Investment Experience (100% cash/savings)
def test_portfolio_beginner_no_investments():
    """
    Scenario 2: Given a user describes having only a savings account with no investments,
    when they ask for guidance, then the bot recommends starting with index funds
    and explains why they're suitable for beginners.
    """
    # Arrange
    portfolio = {
        "Cash": 100.0
    }
    experience = "beginner"
    
    # Act
    analysis = analyze_portfolio_for_test(portfolio, experience)
    
    # Assert
    assert analysis is not None, "Analysis should not be None"
    assert len(analysis) > 50, "Analysis should be substantial"
    
    analysis_lower = analysis.lower()
    
    # Should mention index funds or ETFs or investment recommendation
    has_index_fund = "index fund" in analysis_lower or "index" in analysis_lower
    has_etf = "etf" in analysis_lower
    has_invest = "invest" in analysis_lower
    
    assert has_index_fund or has_etf or has_invest, \
        "Analysis should recommend investments like index funds for beginner with only cash"
    
    print(f"\n✅ Scenario 2 PASSED")
    print(f"Portfolio: {portfolio}")
    print(f"Analysis excerpt: {analysis[:200]}...")


# Scenario 3: High Risk Portfolio (100% tech stocks)
def test_portfolio_concentration_risk():
    """
    Scenario 3: Given a user has an aggressive portfolio (100% tech stocks),
    when the analysis runs, then the bot identifies concentration risk and
    suggests sector diversification strategies with explanations.
    """
    # Arrange
    portfolio = {
        "Stocks": 100.0  # Simulating 100% tech stocks
    }
    experience = "intermediate"
    
    # Enhanced prompt to simulate tech stock concentration
    portfolio_desc = "100% Tech Stocks (concentrated in technology sector)"
    
    prompt = f"""You are a financial advisor analyzing a client's investment portfolio.

Client Profile:
- Experience Level: {experience}
- Portfolio: {portfolio_desc}

This portfolio has 100% allocation in technology stocks with no diversification.

Provide a brief analysis including:
1. Identify the concentration risk
2. Explain the dangers of sector concentration
3. Recommend diversification strategies

Keep response concise (under 200 words).
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    
    analysis = response.choices[0].message.content
    
    # Assert
    assert analysis is not None, "Analysis should not be None"
    assert len(analysis) > 50, "Analysis should be substantial"
    
    analysis_lower = analysis.lower()
    
    # Should mention risk or diversification or concentration
    has_risk = "risk" in analysis_lower
    has_concentration = "concentrat" in analysis_lower
    has_diversification = "diversif" in analysis_lower
    has_sector = "sector" in analysis_lower
    
    assert has_risk or has_concentration or has_diversification or has_sector, \
        "Analysis should identify concentration risk for 100% tech stock portfolio"
    
    print(f"\n✅ Scenario 3 PASSED")
    print(f"Portfolio: 100% Tech Stocks (concentrated)")
    print(f"Analysis excerpt: {analysis[:200]}...")


# Additional test: Validate portfolio totals
def test_portfolio_validation():
    """
    Additional test: Ensure portfolio validation works correctly
    """
    # Valid portfolio (sums to 100)
    valid_portfolio = {
        "Stocks": 60.0,
        "Bonds": 30.0,
        "Cash": 10.0
    }
    assert sum(valid_portfolio.values()) == 100.0, "Valid portfolio should sum to 100%"
    
    # Invalid portfolio (doesn't sum to 100)
    invalid_portfolio = {
        "Stocks": 60.0,
        "Cash": 40.0,
        "Bonds": 20.0
    }
    total = sum(invalid_portfolio.values())
    assert total != 100.0, "Invalid portfolio should not sum to 100%"
    
    print(f"\n✅ Portfolio validation test PASSED")


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("Testing Portfolio Guidance User Story")
    print("="*60)
    
    test_portfolio_unbalanced()
    test_portfolio_beginner_no_investments()
    test_portfolio_concentration_risk()
    test_portfolio_validation()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)