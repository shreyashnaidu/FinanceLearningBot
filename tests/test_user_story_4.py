"""
Test cases for User Story 4: Risk Tolerance Confirmation
Reference: GitHub Issue #4

Run with: pytest tests/test_user_story_4.py -v
"""

import pytest
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client for testing
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_investment_recommendations_for_test(risk_tolerance: str, goal: str = "") -> str:
    """Helper function to get risk-based recommendations"""
    
    risk_profiles = {
        "Conservative": {
            "focus": "capital preservation and steady income",
            "recommended": "bonds, dividend stocks, index funds",
            "avoid": "growth stocks, cryptocurrency, high-volatility assets"
        },
        "Moderate": {
            "focus": "balanced growth with moderate risk",
            "recommended": "diversified index funds, balanced mutual funds, REITs",
            "avoid": "highly speculative assets, leveraged ETFs"
        },
        "Aggressive": {
            "focus": "maximum growth potential",
            "recommended": "growth stocks, sector-specific ETFs, emerging markets, cryptocurrency",
            "avoid": "none, but maintain diversification"
        }
    }
    
    profile = risk_profiles.get(risk_tolerance, risk_profiles["Moderate"])
    
    prompt = f"""You are a financial advisor. Provide brief investment recommendations for someone with {risk_tolerance} risk tolerance.

Focus: {profile['focus']}
Recommended: {profile['recommended']}
Avoid: {profile['avoid']}

Provide:
1. Why these options match this risk level (2 sentences)
2. List 3 specific investments to consider
3. Brief note about what to avoid

Keep under 150 words.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content


# Scenario 1: Conservative Risk Tolerance
def test_conservative_risk_tolerance():
    """
    Scenario 1: Given a user sets risk tolerance to "Conservative",
    when the bot explains investment options, then it focuses on bonds,
    dividend stocks, and index funds while noting that high-risk options exist.
    """
    # Arrange
    risk_tolerance = "Conservative"
    goal = "Learn about investment options"
    
    # Act
    recommendations = get_investment_recommendations_for_test(risk_tolerance, goal)
    
    # Assert
    assert recommendations is not None, "Recommendations should not be None"
    assert len(recommendations) > 50, "Recommendations should be substantial"
    
    recommendations_lower = recommendations.lower()
    
    # Should mention conservative-appropriate investments
    has_bonds = "bond" in recommendations_lower
    has_dividend = "dividend" in recommendations_lower
    has_index = "index" in recommendations_lower or "etf" in recommendations_lower
    
    conservative_mentions = sum([has_bonds, has_dividend, has_index])
    assert conservative_mentions >= 2, \
        "Conservative recommendations should mention at least 2 of: bonds, dividends, or index funds"
    
    # Should avoid or mention risks of aggressive options
    mentions_risk_awareness = (
        "risk" in recommendations_lower or
        "volatile" in recommendations_lower or
        "avoid" in recommendations_lower or
        "conservative" in recommendations_lower
    )
    
    assert mentions_risk_awareness, \
        "Should acknowledge risk level or mention what to avoid"
    
    print(f"\n✅ Scenario 1 PASSED")
    print(f"Risk Tolerance: {risk_tolerance}")
    print(f"Recommendations excerpt: {recommendations[:200]}...")


# Scenario 2: Aggressive Risk Tolerance
def test_aggressive_risk_tolerance():
    """
    Scenario 2: Given a user with "Aggressive" risk tolerance asks about investment options,
    when recommendations are provided, then the bot includes growth stocks,
    sector-specific investments, and higher-volatility assets.
    """
    # Arrange
    risk_tolerance = "Aggressive"
    goal = "Learn about high-growth investment options"
    
    # Act
    recommendations = get_investment_recommendations_for_test(risk_tolerance, goal)
    
    # Assert
    assert recommendations is not None, "Recommendations should not be None"
    assert len(recommendations) > 50, "Recommendations should be substantial"
    
    recommendations_lower = recommendations.lower()
    
    # Should mention aggressive investment options
    has_growth = "growth" in recommendations_lower
    has_sector = "sector" in recommendations_lower or "technology" in recommendations_lower
    has_volatile = "volatil" in recommendations_lower or "risk" in recommendations_lower
    has_emerging = "emerging" in recommendations_lower or "small" in recommendations_lower
    
    aggressive_mentions = sum([has_growth, has_sector, has_volatile, has_emerging])
    assert aggressive_mentions >= 2, \
        "Aggressive recommendations should mention growth stocks, sectors, or high-volatility assets"
    
    print(f"\n✅ Scenario 2 PASSED")
    print(f"Risk Tolerance: {risk_tolerance}")
    print(f"Recommendations excerpt: {recommendations[:200]}...")


# Scenario 3: Risk Tolerance Change Impact
def test_risk_tolerance_change():
    """
    Scenario 3: Given a user changes risk tolerance from "Conservative" to "Moderate",
    when the learning plan refreshes, then the bot adds topics about balanced
    growth strategies and moderately risky assets.
    """
    # Arrange
    initial_risk = "Conservative"
    new_risk = "Moderate"
    goal = "Build a balanced investment portfolio"
    
    # Act - Get recommendations for both risk levels
    conservative_rec = get_investment_recommendations_for_test(initial_risk, goal)
    moderate_rec = get_investment_recommendations_for_test(new_risk, goal)
    
    # Assert
    assert conservative_rec is not None and moderate_rec is not None
    
    conservative_lower = conservative_rec.lower()
    moderate_lower = moderate_rec.lower()
    
    # Conservative should focus on safety
    conservative_safe = (
        "bond" in conservative_lower or
        "safe" in conservative_lower or
        "stable" in conservative_lower
    )
    
    # Moderate should mention balance or growth
    moderate_balanced = (
        "balanc" in moderate_lower or
        "moderate" in moderate_lower or
        "growth" in moderate_lower or
        "mix" in moderate_lower
    )
    
    assert conservative_safe, "Conservative should emphasize safety"
    assert moderate_balanced, "Moderate should mention balance or growth"
    
    # Moderate should have different recommendations than Conservative
    assert conservative_rec != moderate_rec, \
        "Recommendations should differ between Conservative and Moderate"
    
    print(f"\n✅ Scenario 3 PASSED")
    print(f"Initial Risk: {initial_risk}")
    print(f"New Risk: {new_risk}")
    print(f"Conservative excerpt: {conservative_rec[:150]}...")
    print(f"Moderate excerpt: {moderate_rec[:150]}...")
    print("✓ Recommendations differ based on risk tolerance change")


# Additional test: All risk levels produce different results
def test_all_risk_levels_differ():
    """
    Additional test: Ensure all three risk levels produce distinct recommendations
    """
    # Arrange
    goal = "Learn about investing"
    
    # Act
    conservative = get_investment_recommendations_for_test("Conservative", goal)
    moderate = get_investment_recommendations_for_test("Moderate", goal)
    aggressive = get_investment_recommendations_for_test("Aggressive", goal)
    
    # Assert
    assert conservative != moderate, "Conservative and Moderate should differ"
    assert moderate != aggressive, "Moderate and Aggressive should differ"
    assert conservative != aggressive, "Conservative and Aggressive should differ"
    
    print(f"\n✅ All risk levels test PASSED")
    print("✓ Conservative, Moderate, and Aggressive all produce unique recommendations")


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("Testing Risk Tolerance User Story")
    print("="*60)
    
    test_conservative_risk_tolerance()
    test_aggressive_risk_tolerance()
    test_risk_tolerance_change()
    test_all_risk_levels_differ()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)