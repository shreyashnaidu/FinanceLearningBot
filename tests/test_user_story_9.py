"""
Test cases for User Story 9: Portfolio Backtesting
Reference: GitHub Issue #9

As a user who wants to validate investment approaches
I want to backtest proposed portfolio strategies against historical data
So that I can see how my planned allocation would have performed in past market conditions

Run with: pytest tests/test_user_story_9.py -v
"""

import pytest
import app   # your Streamlit app file
from app import backtest_portfolio, compare_backtest_strategies, backtest_crisis_period


# Scenario 1: Basic Portfolio Backtest
def test_portfolio_backtest_10_years():
    """
    Scenario 1: Given a user has a proposed portfolio allocation (70% S&P 500, 20% bonds, 10% gold),
    when they request backtesting, then the bot shows how this portfolio would have performed
    over the past 10 years with annual returns and drawdown periods.
    """
    # Arrange
    portfolio = {
        "S&P 500": 70,
        "Bonds": 20,
        "Gold": 10
    }
    years = 10
    
    # Act
    results = backtest_portfolio(portfolio, years)
    
    # Assert - Check structure
    assert results is not None, "Backtest should return results"
    assert "annual_returns" in results, "Should include annual returns"
    assert "total_return" in results, "Should include total return"
    assert "average_return" in results, "Should include average return"
    assert "max_drawdown" in results, "Should include max drawdown"
    assert "volatility" in results, "Should include volatility"
    assert "sharpe_ratio" in results, "Should include Sharpe ratio"
    assert "years" in results, "Should include years"
    
    # Assert - Check data validity
    assert len(results["annual_returns"]) == years, \
        f"Should have {years} annual returns"
    assert results["years"] == years, "Years should match input"
    assert isinstance(results["total_return"], (int, float)), \
        "Total return should be numeric"
    assert isinstance(results["average_return"], (int, float)), \
        "Average return should be numeric"
    assert isinstance(results["max_drawdown"], (int, float)), \
        "Max drawdown should be numeric"
    assert isinstance(results["sharpe_ratio"], (int, float)), \
        "Sharpe ratio should be numeric"
    
    # Assert - Check reasonable values
    assert -1.0 <= results["max_drawdown"] <= 0.5, \
        "Max drawdown should be between -100% and 50%"
    assert -0.5 <= results["average_return"] <= 0.5, \
        "Average annual return should be reasonable (-50% to 50%)"
    
    print(f"\n✅ Scenario 1 PASSED")
    print(f"Portfolio: {portfolio}")
    print(f"Total Return ({years} years): {results['total_return']:.1%}")
    print(f"Average Annual Return: {results['average_return']:.1%}")
    print(f"Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")


# Scenario 2: Strategy Comparison
def test_compare_strategies_80_20_vs_60_40():
    """
    Scenario 2: Given a user wants to compare two strategies,
    when they request "compare 80/20 vs 60/40 stocks/bonds",
    then the bot presents side-by-side performance metrics
    (total return, volatility, maximum drawdown, Sharpe ratio).
    """
    # Arrange
    strategy1 = {"Stocks": 80, "Bonds": 20}
    strategy2 = {"Stocks": 60, "Bonds": 40}
    label1 = "80/20 Strategy"
    label2 = "60/40 Strategy"
    years = 10
    
    # Act
    comparison = compare_backtest_strategies(
        strategy1, strategy2, label1, label2, years
    )
    
    # Assert - Check output exists and is substantial
    assert comparison is not None, "Comparison should return a result"
    assert len(comparison) > 100, "Comparison should be substantial (>100 chars)"
    
    comparison_lower = comparison.lower()
    
    # Assert - Should mention both strategies
    mentions_strategies = (
        "80" in comparison or "80/20" in comparison or label1.lower() in comparison_lower
    ) and (
        "60" in comparison or "60/40" in comparison or label2.lower() in comparison_lower
    )
    assert mentions_strategies, "Should mention both strategies"
    
    # Assert - Should discuss performance metrics
    has_metrics = (
        "return" in comparison_lower or
        "performance" in comparison_lower or
        "%" in comparison
    )
    assert has_metrics, "Should discuss return/performance metrics"
    
    # Assert - Should discuss risk
    has_risk = (
        "risk" in comparison_lower or
        "volatil" in comparison_lower or
        "drawdown" in comparison_lower or
        "sharpe" in comparison_lower
    )
    assert has_risk, "Should discuss risk metrics"
    
    # Assert - Should provide guidance
    has_guidance = (
        "better" in comparison_lower or
        "suit" in comparison_lower or
        "investor" in comparison_lower or
        "recommend" in comparison_lower or
        "appropriate" in comparison_lower
    )
    assert has_guidance, "Should provide investor guidance"
    
    print(f"\n✅ Scenario 2 PASSED")
    print(f"Strategy 1: {strategy1}")
    print(f"Strategy 2: {strategy2}")
    print(f"Comparison excerpt: {comparison[:200]}...")


# Scenario 3: Crisis Period Performance
def test_crisis_performance_2008():
    """
    Scenario 3: Given a user asks "how would my portfolio have done during the 2008 crisis?",
    when the bot responds, then it shows the specific performance during that period
    and explains what factors contributed to gains or losses.
    """
    # Arrange
    portfolio = {
        "Stocks": 60,
        "Bonds": 30,
        "Gold": 10
    }
    crisis = "2008 financial crisis"
    
    # Act
    analysis = backtest_crisis_period(portfolio, crisis)
    
    # Assert - Check output exists
    assert analysis is not None, "Crisis analysis should return a result"
    assert len(analysis) > 80, "Crisis analysis should be substantial"
    
    analysis_lower = analysis.lower()
    
    # Assert - Should mention the crisis
    mentions_crisis = (
        "2008" in analysis or
        "financial crisis" in analysis_lower or
        "crisis" in analysis_lower
    )
    assert mentions_crisis, "Should mention the 2008 financial crisis"
    
    # Assert - Should discuss portfolio performance
    has_performance = (
        "portfolio" in analysis_lower or
        "%" in analysis or
        "return" in analysis_lower or
        "loss" in analysis_lower or
        "gain" in analysis_lower or
        "perform" in analysis_lower
    )
    assert has_performance, "Should discuss portfolio performance"
    
    # Assert - Should mention different asset classes
    asset_mentions = [
        "stock" in analysis_lower,
        "bond" in analysis_lower,
        "gold" in analysis_lower,
    ]
    assets_discussed = sum(asset_mentions)
    assert assets_discussed >= 2, "Should discuss at least 2 asset classes"
    
    # Assert - Should provide lessons/explanation
    has_lessons = (
        "lesson" in analysis_lower or
        "protect" in analysis_lower or
        "safe" in analysis_lower or
        "because" in analysis_lower or
        "explain" in analysis_lower or
        "factor" in analysis_lower or
        "contributed" in analysis_lower
    )
    assert has_lessons, "Should explain factors or provide lessons"
    
    print(f"\n✅ Scenario 3 PASSED")
    print(f"Portfolio: {portfolio}")
    print(f"Crisis: {crisis}")
    print(f"Analysis excerpt: {analysis[:200]}...")


# Additional Test: COVID-19 Crisis
def test_crisis_performance_covid():
    """Test backtesting with COVID-19 crash"""
    # Arrange
    portfolio = {"Stocks": 70, "Bonds": 20, "Cash": 10}
    crisis = "COVID-19 crash"
    
    # Act
    analysis = backtest_crisis_period(portfolio, crisis)
    
    # Assert
    assert analysis is not None
    assert len(analysis) > 50
    
    analysis_lower = analysis.lower()
    mentions_covid = "covid" in analysis_lower or "pandemic" in analysis_lower
    assert mentions_covid, "Should mention COVID-19"
    
    print(f"\n✅ COVID-19 crisis test PASSED")
    print(f"Analysis excerpt: {analysis[:150]}...")


# Additional Test: Dot-com Bubble
def test_crisis_performance_dotcom():
    """Test backtesting with dot-com bubble"""
    # Arrange
    portfolio = {"Stocks": 100}
    crisis = "dot-com bubble"
    
    # Act
    analysis = backtest_crisis_period(portfolio, crisis)
    
    # Assert
    assert analysis is not None
    assert len(analysis) > 50
    
    analysis_lower = analysis.lower()
    mentions_dotcom = (
        "dot-com" in analysis_lower or
        "tech" in analysis_lower or
        "bubble" in analysis_lower
    )
    assert mentions_dotcom, "Should mention dot-com or tech bubble"
    
    print(f"\n✅ Dot-com bubble test PASSED")
    print(f"Analysis excerpt: {analysis[:150]}...")


# Additional Test: Multiple Year Periods
def test_backtest_different_periods():
    """Test backtesting over different time periods (5, 10, 20, 30 years)"""
    # Arrange
    portfolio = {"Stocks": 60, "Bonds": 40}
    periods = [5, 10, 20, 30]
    
    for years in periods:
        # Act
        results = backtest_portfolio(portfolio, years)
        
        # Assert
        assert results is not None
        assert len(results["annual_returns"]) == years
        assert results["years"] == years
        
        print(f"✅ {years}-year backtest passed")
    
    print(f"\n✅ Multiple period test PASSED")
    print(f"Tested periods: {periods}")


# Additional Test: Different Portfolio Compositions
def test_backtest_various_portfolios():
    """Test backtesting with various portfolio compositions"""
    # Arrange
    portfolios = [
        {"Stocks": 100},  # All stocks
        {"Bonds": 100},   # All bonds
        {"Stocks": 60, "Bonds": 40},  # Balanced
        {"Stocks": 70, "Bonds": 20, "Gold": 10},  # Diversified
        {"S&P 500": 50, "Bonds": 30, "Real Estate": 20},  # Mixed
    ]
    
    for portfolio in portfolios:
        # Act
        results = backtest_portfolio(portfolio, 10)
        
        # Assert
        assert results is not None
        assert "total_return" in results
        assert "sharpe_ratio" in results
        
        print(f"✅ Backtest passed for: {portfolio}")
    
    print(f"\n✅ Various portfolio test PASSED")
    print(f"Tested {len(portfolios)} different portfolio compositions")


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("Testing Portfolio Backtesting User Story")
    print("="*60)
    
    test_portfolio_backtest_10_years()
    test_compare_strategies_80_20_vs_60_40()
    test_crisis_performance_2008()
    test_crisis_performance_covid()
    test_crisis_performance_dotcom()
    test_backtest_different_periods()
    test_backtest_various_portfolios()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)