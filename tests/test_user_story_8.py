"""
Test cases for User Story 8: Historical Data and Empirical Trends
Reference: GitHub Issue #8

Run with: pytest tests/test_user_story_8.py -v
"""

import pytest
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client for testing
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_historical_performance_for_test(asset_class: str) -> str:
    """Get historical performance data for an asset class"""
    
    prompt = f"""Provide historical performance data for {asset_class}.

Include:
1. Average annual returns over 10, 20, and 30 years
2. Brief context about market cycles that affected performance
3. Notable periods of high/low performance

Keep response under 200 words. Use approximate historical data.

Example format:
"S&P 500 index funds have delivered:
- 10-year average: ~12% annually
- 20-year average: ~10% annually  
- 30-year average: ~10% annually

Context: Performance includes the 2008 financial crisis recovery and COVID-19 pandemic bounce-back..."
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()


def compare_allocation_performance_for_test(allocation1: str, allocation2: str) -> str:
    """Compare historical performance of different allocations"""
    
    prompt = f"""Compare historical performance of these portfolios:

Portfolio 1: {allocation1}
Portfolio 2: {allocation2}

Provide:
1. Historical average annual returns for each
2. Volatility comparison (which had bigger swings)
3. Which performed better in different market conditions

Keep under 200 words. Use approximate historical data.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=350
    )
    
    return response.choices[0].message.content.strip()


def get_market_event_data_for_test(event: str) -> str:
    """Get data about how assets performed during major market events"""
    
    prompt = f"""Describe how different asset classes performed during the {event}.

Include:
1. What happened during this event
2. How stocks, bonds, and cash performed
3. Key lessons for investors

Keep under 200 words. Focus on educational value.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()


# Scenario 1: Historical Performance Query
def test_historical_performance_query():
    """
    Scenario 1: Given a user asks "How have index funds performed historically?",
    when the bot responds, then it provides average annual returns over the past
    10, 20, and 30 years with context about market cycles.
    """
    # Arrange
    asset_class = "index funds"
    
    # Act
    performance_data = get_historical_performance_for_test(asset_class)
    
    # Assert
    assert performance_data is not None
    assert len(performance_data) > 100, "Response should be substantial"
    
    performance_lower = performance_data.lower()
    
    # Should mention multiple timeframes
    has_10_year = "10" in performance_data or "ten" in performance_lower
    has_20_year = "20" in performance_data or "twenty" in performance_lower
    has_30_year = "30" in performance_data or "thirty" in performance_lower
    
    timeframes_mentioned = sum([has_10_year, has_20_year, has_30_year])
    assert timeframes_mentioned >= 2, \
        "Should mention at least 2 different timeframes (10, 20, or 30 years)"
    
    # Should include returns data
    has_returns = (
        "return" in performance_lower or
        "%" in performance_data or
        "percent" in performance_lower or
        "annual" in performance_lower
    )
    assert has_returns, "Should mention returns or percentages"
    
    # Should provide context
    has_context = (
        "market" in performance_lower or
        "cycle" in performance_lower or
        "crisis" in performance_lower or
        "period" in performance_lower
    )
    assert has_context, "Should provide market context"
    
    print(f"\n✅ Scenario 1 PASSED")
    print(f"Asset Class: {asset_class}")
    print(f"Response excerpt: {performance_data[:200]}...")


# Scenario 2: Portfolio Allocation Evidence
def test_portfolio_allocation_evidence():
    """
    Scenario 2: Given the bot recommends a 60/40 stock/bond portfolio,
    when the user requests evidence, then the bot shows historical performance
    data comparing this allocation to alternatives (100% stocks, 80/20, etc.).
    """
    # Arrange
    recommended_allocation = "60% stocks / 40% bonds"
    alternative_allocation = "100% stocks"
    
    # Act
    comparison = compare_allocation_performance_for_test(
        recommended_allocation, alternative_allocation
    )
    
    # Assert
    assert comparison is not None
    assert len(comparison) > 100, "Comparison should be substantial"
    
    comparison_lower = comparison.lower()
    
    # Should mention both allocations
    mentions_60_40 = "60" in comparison and "40" in comparison
    mentions_100_stocks = "100" in comparison or "all stock" in comparison_lower
    
    assert mentions_60_40 or "first" in comparison_lower, \
        "Should reference the 60/40 allocation"
    assert mentions_100_stocks or "second" in comparison_lower, \
        "Should reference the 100% stocks allocation"
    
    # Should discuss performance differences
    has_performance = (
        "return" in comparison_lower or
        "performance" in comparison_lower or
        "%" in comparison
    )
    assert has_performance, "Should discuss performance metrics"
    
    # Should discuss risk/volatility
    has_risk_discussion = (
        "risk" in comparison_lower or
        "volatil" in comparison_lower or
        "swing" in comparison_lower or
        "stable" in comparison_lower
    )
    assert has_risk_discussion, "Should discuss risk or volatility differences"
    
    print(f"\n✅ Scenario 2 PASSED")
    print(f"Allocation 1: {recommended_allocation}")
    print(f"Allocation 2: {alternative_allocation}")
    print(f"Comparison excerpt: {comparison[:200]}...")


# Scenario 3: Market Event Examples
def test_market_event_examples():
    """
    Scenario 3: Given a user is learning about market volatility,
    when they ask for examples, then the bot presents data from major
    market events (2008 crisis, COVID crash) showing how different
    asset classes behaved.
    """
    # Arrange
    market_events = ["2008 financial crisis", "COVID-19 crash"]
    
    for event in market_events:
        # Act
        event_data = get_market_event_data_for_test(event)
        
        # Assert
        assert event_data is not None
        assert len(event_data) > 80, f"Event data for {event} should be substantial"
        
        event_lower = event_data.lower()
        
        # Should mention the specific event
        event_terms = event.lower().split()
        mentions_event = any(term in event_lower for term in event_terms)
        assert mentions_event, f"Should mention the {event}"
        
        # Should discuss different asset classes
        asset_mentions = [
            "stock" in event_lower,
            "bond" in event_lower,
            "cash" in event_lower or "treasury" in event_lower,
        ]
        assets_discussed = sum(asset_mentions)
        assert assets_discussed >= 2, \
            f"Should discuss at least 2 asset classes for {event}"
        
        # Should provide educational context
        has_education = (
            "lesson" in event_lower or
            "learn" in event_lower or
            "important" in event_lower or
            "show" in event_lower or
            "demonstrate" in event_lower
        )
        assert has_education, f"Should provide educational context for {event}"
        
        print(f"\n✅ Event test passed: {event}")
        print(f"Data excerpt: {event_data[:150]}...")
    
    print(f"\n✅ Scenario 3 PASSED")
    print(f"Tested {len(market_events)} major market events")


# Additional test: Data should be educational, not just numbers
def test_data_includes_context():
    """Ensure historical data includes educational context, not just numbers"""
    
    # Act
    data = get_historical_performance_for_test("S&P 500")
    
    # Assert
    data_lower = data.lower()
    
    # Should not be just numbers - should explain
    has_explanation = (
        "because" in data_lower or
        "due to" in data_lower or
        "context" in data_lower or
        "period" in data_lower or
        "during" in data_lower or
        "cycle" in data_lower
    )
    
    assert has_explanation, \
        "Historical data should include explanatory context, not just numbers"
    
    print(f"\n✅ Context test PASSED")
    print("Historical data includes educational context")


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("Testing Historical Data User Story")
    print("="*60)
    
    test_historical_performance_query()
    test_portfolio_allocation_evidence()
    test_market_event_examples()
    test_data_includes_context()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)