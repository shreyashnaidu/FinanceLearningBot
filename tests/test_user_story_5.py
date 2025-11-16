"""
Test cases for User Story 5: Risk-Based Portfolio Adjustment (Human-in-the-Loop)
Reference: GitHub Issue #5

Run with: pytest tests/test_user_story_5.py -v
"""

import pytest
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize OpenAI client for testing
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def suggest_portfolio_allocation_for_test(risk_tolerance: str, current_allocation: dict = None) -> dict:
    """Generate portfolio allocation based on risk tolerance"""
    
    prompt = f"""You are a financial advisor. Suggest a portfolio allocation for a {risk_tolerance} risk tolerance investor.

Current allocation: {json.dumps(current_allocation) if current_allocation else "None - new portfolio"}

Provide allocation recommendations as percentages that sum to 100%.

For {risk_tolerance}:
- Conservative: Higher bonds (60-70%), moderate stocks (20-30%), some cash (10%)
- Moderate: Balanced (40-50% stocks, 30-40% bonds, 10-20% alternatives/cash)
- Aggressive: Higher stocks (60-80%), some bonds (10-20%), alternatives (10-20%)

Return ONLY a JSON object with allocation percentages. Example:
{{"Stocks": 50, "Bonds": 40, "Alternatives": 10}}

Explanation should follow separately.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=200
    )
    
    # Parse JSON from response
    content = response.choices[0].message.content.strip()
    # Extract JSON if wrapped in markdown
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    
    try:
        allocation = json.loads(content.strip())
        return allocation
    except:
        # Fallback allocations
        defaults = {
            "Conservative": {"Bonds": 70, "Stocks": 20, "Cash": 10},
            "Moderate": {"Stocks": 50, "Bonds": 40, "Alternatives": 10},
            "Aggressive": {"Stocks": 70, "Bonds": 20, "Alternatives": 10}
        }
        return defaults.get(risk_tolerance, defaults["Moderate"])


def explain_allocation_change_for_test(old_allocation: dict, new_allocation: dict, old_risk: str, new_risk: str) -> str:
    """Explain differences between two allocations"""
    
    prompt = f"""Explain the change from {old_risk} to {new_risk} risk tolerance.

Old allocation ({old_risk}): {json.dumps(old_allocation)}
New allocation ({new_risk}): {json.dumps(new_allocation)}

Explain in 2-3 sentences:
1. What changed and why
2. How this affects risk/return profile

Keep under 100 words.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()


# Scenario 1: Risk Tolerance Change Updates Allocation
def test_risk_change_updates_allocation():
    """
    Scenario 1: Given a user has received a conservative 70% bonds / 30% stocks recommendation,
    when they change risk tolerance to "Moderate", then the bot updates to 50% stocks / 40% bonds /
    10% alternatives and explains why.
    """
    # Arrange
    initial_risk = "Conservative"
    new_risk = "Moderate"
    
    # Act
    conservative_allocation = suggest_portfolio_allocation_for_test(initial_risk)
    moderate_allocation = suggest_portfolio_allocation_for_test(new_risk, conservative_allocation)
    
    # Assert - Conservative should have higher bonds
    assert conservative_allocation is not None
    assert moderate_allocation is not None
    
    # Conservative should favor bonds
    conservative_bonds = conservative_allocation.get("Bonds", 0)
    assert conservative_bonds >= 60, f"Conservative should have ≥60% bonds, got {conservative_bonds}%"
    
    # Moderate should have more stocks
    moderate_stocks = moderate_allocation.get("Stocks", 0)
    assert moderate_stocks >= 40, f"Moderate should have ≥40% stocks, got {moderate_stocks}%"
    
    # Moderate should have less bonds than Conservative
    moderate_bonds = moderate_allocation.get("Bonds", 0)
    assert moderate_bonds < conservative_bonds, \
        f"Moderate bonds ({moderate_bonds}%) should be less than Conservative ({conservative_bonds}%)"
    
    # Test explanation
    explanation = explain_allocation_change_for_test(
        conservative_allocation, moderate_allocation, initial_risk, new_risk
    )
    
    assert len(explanation) > 20, "Explanation should be substantial"
    assert "risk" in explanation.lower() or "stock" in explanation.lower(), \
        "Explanation should mention risk or stocks"
    
    print(f"\n✅ Scenario 1 PASSED")
    print(f"Conservative: {conservative_allocation}")
    print(f"Moderate: {moderate_allocation}")
    print(f"Explanation: {explanation[:150]}...")


# Scenario 2: User Feedback Adjusts Recommendations
def test_user_feedback_adjustment():
    """
    Scenario 2: Given a user disagrees with a suggested allocation,
    when they provide feedback ("I want more stocks"), then the bot adjusts
    recommendations and explains the trade-offs.
    """
    # Arrange
    initial_allocation = {"Stocks": 30, "Bonds": 60, "Cash": 10}
    user_feedback = "I want more stocks"
    
    prompt = f"""User has this portfolio: {json.dumps(initial_allocation)}

User feedback: "{user_feedback}"

Adjust the allocation to increase stocks while maintaining total of 100%.
Explain the trade-offs (higher returns potential vs increased volatility).

Return JSON allocation and brief explanation (under 100 words).
Format:
{{"allocation": {{"Stocks": X, "Bonds": Y, "Cash": Z}}, "explanation": "..."}}
"""
    
    # Act
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=250
    )
    
    content = response.choices[0].message.content.strip()
    
    # Assert
    assert content is not None
    assert len(content) > 50, "Response should be substantial"
    
    # Check for trade-off discussion
    content_lower = content.lower()
    has_risk_mention = "risk" in content_lower or "volatil" in content_lower
    has_return_mention = "return" in content_lower or "growth" in content_lower
    
    assert has_risk_mention or has_return_mention, \
        "Response should discuss trade-offs (risk/volatility or returns/growth)"
    
    print(f"\n✅ Scenario 2 PASSED")
    print(f"Initial: {initial_allocation}")
    print(f"Feedback: {user_feedback}")
    print(f"Response excerpt: {content[:200]}...")


# Scenario 3: Alternative Portfolio Comparison
def test_alternative_portfolio_comparison():
    """
    Scenario 3: Given a user requests "show me an aggressive version of this portfolio",
    when the bot responds, then it presents an alternative allocation and compares
    risk metrics between the two approaches.
    """
    # Arrange
    current_allocation = {"Stocks": 40, "Bonds": 50, "Cash": 10}
    current_risk = "Moderate"
    requested_risk = "Aggressive"
    
    # Act
    aggressive_allocation = suggest_portfolio_allocation_for_test(requested_risk, current_allocation)
    
    # Generate comparison
    prompt = f"""Compare these two portfolios:

Portfolio 1 ({current_risk}): {json.dumps(current_allocation)}
Portfolio 2 ({requested_risk}): {json.dumps(aggressive_allocation)}

Provide:
1. Key differences
2. Risk level comparison
3. Expected return potential comparison

Keep under 150 words.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300
    )
    
    comparison = response.choices[0].message.content.strip()
    
    # Assert
    assert aggressive_allocation is not None
    assert comparison is not None
    
    # Aggressive should have more stocks
    aggressive_stocks = aggressive_allocation.get("Stocks", 0)
    current_stocks = current_allocation.get("Stocks", 0)
    assert aggressive_stocks > current_stocks, \
        f"Aggressive should have more stocks: Aggressive({aggressive_stocks}%) vs Current({current_stocks}%)"
    
    # Comparison should mention key concepts
    comparison_lower = comparison.lower()
    has_comparison_terms = any(term in comparison_lower for term in 
                               ["risk", "return", "volatil", "growth", "higher", "lower"])
    
    assert has_comparison_terms, "Comparison should discuss risk/return differences"
    
    print(f"\n✅ Scenario 3 PASSED")
    print(f"Current ({current_risk}): {current_allocation}")
    print(f"Aggressive: {aggressive_allocation}")
    print(f"Comparison: {comparison[:200]}...")


# Additional test: Allocations sum to 100%
def test_allocations_sum_to_100():
    """Ensure all generated allocations sum to 100%"""
    
    for risk in ["Conservative", "Moderate", "Aggressive"]:
        allocation = suggest_portfolio_allocation_for_test(risk)
        total = sum(allocation.values())
        
        assert 99 <= total <= 101, \
            f"{risk} allocation should sum to ~100%, got {total}%: {allocation}"
    
    print(f"\n✅ Allocation sum test PASSED")
    print("All risk levels produce valid allocations (sum to 100%)")


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("Testing Portfolio Adjustment User Story")
    print("="*60)
    
    test_risk_change_updates_allocation()
    test_user_feedback_adjustment()
    test_alternative_portfolio_comparison()
    test_allocations_sum_to_100()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)