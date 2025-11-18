import os
import re
import json
import pickle
import base64
from typing import TypedDict, List, Dict
from html import escape
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

# ============= Session & UI Helpers =============


def init_session_state():
    ss = st.session_state
    ss.setdefault("goal_state", None)
    ss.setdefault("explanation", "")
    ss.setdefault("glossary", [])
    ss.setdefault("portfolio_analysis_only", None)
    ss.setdefault("investment_recommendations", None)
    ss.setdefault("suggested_allocation", None)  # User Story 5
    ss.setdefault("portfolio_comparison", None)  # User Story 5
    ss.setdefault("allocation_history", [])      # User Story 5
    ss.setdefault("reasoning_answer", "")        # User Story 6
    ss.setdefault("historical_data_display", None)
    ss.setdefault("backtest_results", None)      # User Story 9
    ss.setdefault("strategy_comparison", None)   # User Story 9
    ss.setdefault("crisis_backtest", None)       # User Story 9
    ss.setdefault("saved_portfolios", [])  # User Story 10
    ss.setdefault("load_message", "")      # User Story 10


def render_explanation_card(text: str, term: str | None = None):
    safe = escape(text).replace("\n", "<br>")
    tag = f" ({escape(term)})" if term else ""
    st.markdown(
        f"""
        <div style="
            background: #ffffff10;
            border: 1px solid rgba(255,255,255,0.25);
            padding: 12px 16px;
            border-radius: 10px;
            margin-top: 8px;
        ">
            <b>🧠 Explanation{tag}</b><br>
            <span style="line-height:1.5;">{safe}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============= OpenAI Client Setup =============

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found! Please set your OPENAI_API_KEY in the .env file")
    st.stop()

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"❌ Error initializing OpenAI: {str(e)}")
    st.stop()


def explain_finance_term(term: str) -> str:
    prompt = f"""
Explain this finance concept simply for a beginner:
"{term}"

- Use plain English
- Give a short real world example
- Keep under 5 sentences
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def get_investment_recommendations(risk_tolerance: str, goal: str = "") -> str:
    """Get risk-appropriate investment recommendations"""

    risk_profiles = {
        "Conservative": {
            "focus": "capital preservation and steady income",
            "recommended": "government bonds, high-grade corporate bonds, dividend-paying blue-chip stocks, index funds (S&P 500, total market), money market funds, and certificates of deposit (CDs)",
            "avoid": "individual growth stocks, cryptocurrency, options/derivatives, sector-specific ETFs, and high-volatility assets",
            "note": "While higher-risk options exist (like growth stocks and crypto), they don't align with your conservative approach and could jeopardize your capital.",
        },
        "Moderate": {
            "focus": "balanced growth with moderate risk",
            "recommended": "diversified index funds, balanced mutual funds (60/40 stocks/bonds), dividend growth stocks, real estate investment trusts (REITs), target-date retirement funds, and a mix of growth and value stocks",
            "avoid": "highly speculative assets, leveraged ETFs, penny stocks, and concentrated sector bets",
            "note": "You can consider some growth opportunities while maintaining a defensive position with bonds and stable dividend stocks.",
        },
        "Aggressive": {
            "focus": "maximum growth potential accepting higher volatility",
            "recommended": "growth stocks (especially in technology and emerging sectors), sector-specific ETFs, small-cap and mid-cap stocks, international emerging markets, some cryptocurrency exposure (5-10% max), and actively managed growth funds",
            "avoid": "nothing is off limits, but maintain some diversification",
            "note": "With aggressive tolerance, you can pursue high-growth opportunities, but remember to diversify across sectors and asset classes to manage downside risk.",
        },
    }

    profile = risk_profiles.get(risk_tolerance, risk_profiles["Moderate"])

    prompt = f"""You are a financial advisor providing investment recommendations.

Risk Tolerance: {risk_tolerance}
Goal: {goal if goal else "general investment education"}

Based on a {risk_tolerance} risk tolerance (focus: {profile['focus']}), provide a brief educational overview of appropriate investment options.

RECOMMENDED for {risk_tolerance} investors: {profile['recommended']}

AVOID or minimize: {profile['avoid']}

IMPORTANT NOTE: {profile['note']}

Provide:
1. A brief explanation of why these options match this risk level (2-3 sentences)
2. List 3-4 specific investment types to learn about
3. A cautionary note about risk-inappropriate options

Keep response under 200 words. Be educational and encouraging.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"


# ============= User Story 5: Allocation Helpers =============


def suggest_portfolio_allocation(risk_tolerance: str, current_allocation: dict | None = None) -> dict:
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

Do not include any markdown formatting, just the JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        allocation = json.loads(content.strip())
        return allocation
    except Exception:
        defaults = {
            "Conservative": {"Bonds": 70, "Stocks": 20, "Cash": 10},
            "Moderate": {"Stocks": 50, "Bonds": 40, "Alternatives": 10},
            "Aggressive": {"Stocks": 70, "Bonds": 20, "Alternatives": 10},
        }
        return defaults.get(risk_tolerance, defaults["Moderate"])


def explain_allocation_change(
    old_allocation: dict,
    new_allocation: dict,
    old_risk: str,
    new_risk: str,
) -> str:
    """Explain differences between two allocations"""

    prompt = f"""Explain the change from {old_risk} to {new_risk} risk tolerance.

Old allocation ({old_risk}): {json.dumps(old_allocation)}
New allocation ({new_risk}): {json.dumps(new_allocation)}

Explain in 2-3 sentences:
1. What changed and why
2. How this affects risk/return profile
3. Key trade-offs

Keep under 150 words.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"


def adjust_allocation_by_feedback(current_allocation: dict, user_feedback: str) -> tuple[dict, str]:
    """Adjust portfolio based on user feedback"""

    prompt = f"""User has this portfolio: {json.dumps(current_allocation)}

User feedback: "{user_feedback}"

Adjust the allocation based on their feedback while maintaining total of 100%.
Explain the trade-offs (higher returns potential vs increased volatility/risk).

Return a JSON object with this structure:
{{
  "allocation": {{"Stocks": X, "Bonds": Y, ...}},
  "explanation": "Brief explanation of changes and trade-offs"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=350,
        )

        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content.strip())
        return result.get("allocation", current_allocation), result.get("explanation", "")
    except Exception as e:
        return current_allocation, f"Could not process feedback: {str(e)}"


def compare_portfolios(
    portfolio1: dict,
    portfolio2: dict,
    label1: str,
    label2: str,
) -> str:
    """Compare two portfolio allocations"""

    prompt = f"""Compare these two portfolios:

Portfolio 1 ({label1}): {json.dumps(portfolio1)}
Portfolio 2 ({label2}): {json.dumps(portfolio2)}

Provide:
1. Key differences in allocation
2. Risk level comparison
3. Expected return potential comparison
4. Which might be suitable for different investor types

Keep under 200 words.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to compare portfolios: {str(e)}"


def get_historical_performance(asset_class: str) -> str:
    """Get historical performance data for an asset class"""
    
    # Historical data knowledge base
    historical_data = {
        "index funds": {
            "10_year": "~12%",
            "20_year": "~10%",
            "30_year": "~10%",
            "context": "Includes recovery from 2008 financial crisis and COVID-19 pandemic bounce-back. Strong long-term growth despite short-term volatility."
        },
        "s&p 500": {
            "10_year": "~12%",
            "20_year": "~10%",
            "30_year": "~10%",
            "context": "Major US stock index showing consistent long-term growth. Experienced significant drops in 2000-2002 (dot-com), 2008 (financial crisis), and 2020 (COVID)."
        },
        "bonds": {
            "10_year": "~3-4%",
            "20_year": "~4-5%",
            "30_year": "~5-6%",
            "context": "Lower but steadier returns than stocks. Performed well during stock market crashes, providing portfolio stability."
        },
        "stocks": {
            "10_year": "~11-13%",
            "20_year": "~9-11%",
            "30_year": "~10%",
            "context": "Higher volatility but strong long-term returns. Best performance over extended periods despite periodic crashes."
        }
    }
    
    # Normalize input
    asset_key = asset_class.lower().strip()
    
    # Get data or use generic response
    if asset_key in historical_data:
        data = historical_data[asset_key]
        base_response = f"""**Historical Performance of {asset_class.title()}:**

**Returns:**
- **10-year average:** {data['10_year']} annually
- **20-year average:** {data['20_year']} annually
- **30-year average:** {data['30_year']} annually

**Context:** {data['context']}

*Note: Past performance doesn't guarantee future results. These are approximate historical averages.*"""
    else:
        # Use AI for assets not in knowledge base
        prompt = f"""Provide historical performance data for {asset_class}.

Include:
1. Average annual returns over 10, 20, and 30 years
2. Brief context about market cycles that affected performance
3. Notable periods of high/low performance

Keep response under 200 words. Use approximate historical data.
Add disclaimer about past performance.
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=300
            )
            base_response = response.choices[0].message.content.strip()
        except Exception as e:
            base_response = f"Unable to retrieve historical data for {asset_class}: {str(e)}"
    
    return base_response


def compare_allocation_performance(allocation1_desc: str, allocation2_desc: str) -> str:
    """Compare historical performance of different portfolio allocations"""
    
    prompt = f"""Compare historical performance of these portfolios:

**Portfolio 1:** {allocation1_desc}
**Portfolio 2:** {allocation2_desc}

Provide:
1. Historical average annual returns for each (use approximate data)
2. Volatility comparison (which had bigger swings)
3. Which performed better in different market conditions (bull markets, bear markets, recessions)
4. Risk/return trade-off analysis

Keep under 250 words. Be educational and use historical evidence.
Include disclaimer about past performance.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to compare allocations: {str(e)}"


def get_market_event_data(event: str) -> str:
    """Get data about how assets performed during major market events"""
    
    # Known market events
    market_events = {
        "2008 financial crisis": {
            "description": "Global financial crisis triggered by mortgage-backed securities collapse",
            "stocks": "S&P 500 fell ~37% in 2008, recovered within 5 years",
            "bonds": "US Treasury bonds gained value as safe haven, +5-10%",
            "lessons": "Diversification helped. All-stock portfolios suffered most. Bonds provided stability. Long-term investors who held recovered."
        },
        "covid-19 crash": {
            "description": "Pandemic-driven market crash in March 2020",
            "stocks": "S&P 500 dropped ~34% in one month, but recovered to new highs within 6 months",
            "bonds": "Treasury bonds remained stable, some gains",
            "lessons": "Fastest crash and fastest recovery in history. Stay-at-home stocks thrived. Market recovery defied economic downturn."
        },
        "dot-com bubble": {
            "description": "Tech stock bubble burst in 2000-2002",
            "stocks": "Nasdaq fell ~78% from peak, S&P 500 fell ~49%",
            "bonds": "Bonds outperformed as investors fled to safety",
            "lessons": "Speculation in unproven companies dangerous. Diversification critical. Tech concentration very risky."
        }
    }
    
    event_lower = event.lower().strip()
    
    # Find matching event
    matched_event = None
    for key in market_events:
        if key in event_lower or event_lower in key:
            matched_event = market_events[key]
            break
    
    if matched_event:
        response = f"""**{event.title()}**

**What Happened:** {matched_event['description']}

**Asset Performance:**
- **Stocks:** {matched_event['stocks']}
- **Bonds:** {matched_event['bonds']}

**Key Lessons:** {matched_event['lessons']}

*Historical data shows the importance of diversification and long-term perspective during market crises.*"""
    else:
        # Use AI for events not in knowledge base
        prompt = f"""Describe how different asset classes performed during the {event}.

Include:
1. What happened during this event
2. How stocks, bonds, and cash performed (use percentages if possible)
3. Key lessons for investors

Keep under 200 words. Focus on educational value.
"""
        
        try:
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=300
            )
            response = ai_response.choices[0].message.content.strip()
        except Exception as e:
            response = f"Unable to retrieve data for {event}: {str(e)}"
    
    return response


# ============= NEW: User Story 6 – Explain "Why?" =============


def explain_recommendation(user_question: str, **kwargs) -> str:
    """
    User Story 6: Explain *why* a recommendation was made.
    """
    risk_tolerance = kwargs.get("risk_tolerance", "not provided")
    age = kwargs.get("age", "not provided")
    time_available = kwargs.get("time_available", "not provided")
    experience = kwargs.get("experience", "not provided")
    
    suggested_allocation = kwargs.get("suggested_allocation")
    investment_recs = kwargs.get("investment_recs")
    goal_state = kwargs.get("goal_state")

    # Build context from available data
    context_parts = []
    
    context_parts.append(f"User Profile: Age {age}, {risk_tolerance} risk tolerance, {experience} experience, {time_available} available per week")
    
    if suggested_allocation:
        alloc = suggested_allocation.get("allocation", {})
        context_parts.append(f"Suggested Allocation: {alloc}")
    
    if investment_recs:
        recs = investment_recs.get("recommendations", "")[:200]
        context_parts.append(f"Investment Recommendations: {recs}")
    
    if goal_state and goal_state.get("subtasks"):
        tasks = [t["task"] for t in goal_state["subtasks"][:3]]
        context_parts.append(f"Learning Path: {', '.join(tasks)}")

    context = "\n".join(context_parts)

    prompt = f"""A user is asking WHY a financial recommendation was made.

User question: "{user_question}"

Context:
{context}

Your task:
- Explain the reasoning behind the recommendation
- Connect reasoning to age, risk tolerance, diversification, and time horizon
- Be concise and educational for beginners
- Keep answer under 6 sentences
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Unable to explain recommendation: {str(e)}"


# ============= User Story 9: Backtesting =============


def backtest_portfolio(allocation: dict, years: int = 10) -> dict:
    """
    Backtest a portfolio allocation over historical period.
    Returns performance metrics including returns and drawdowns.
    """
    # Historical average returns (approximate)
    historical_returns = {
        "S&P 500": 0.12,
        "Stocks": 0.11,
        "Bonds": 0.04,
        "Gold": 0.05,
        "Cash": 0.02,
        "Real Estate": 0.08,
        "Cryptocurrency": 0.25,  # High variance
        "Alternatives": 0.07
    }
    
    # Normalize allocation keys
    normalized_allocation = {}
    for key, value in allocation.items():
        # Map common variations to standard keys
        normalized_key = key
        if "stock" in key.lower() and "s&p" not in key.lower():
            normalized_key = "Stocks"
        elif "s&p" in key.lower() or "index" in key.lower():
            normalized_key = "S&P 500"
        elif "bond" in key.lower():
            normalized_key = "Bonds"
        elif "gold" in key.lower():
            normalized_key = "Gold"
        elif "cash" in key.lower():
            normalized_key = "Cash"
        elif "real estate" in key.lower() or "reit" in key.lower():
            normalized_key = "Real Estate"
        elif "crypto" in key.lower():
            normalized_key = "Cryptocurrency"
        
        normalized_allocation[normalized_key] = value
    
    # Calculate weighted average return
    annual_returns = []
    for year in range(years):
        year_return = 0
        for asset, pct in normalized_allocation.items():
            base_return = historical_returns.get(asset, 0.07)
            # Add some volatility simulation
            import random
            volatility = random.uniform(-0.15, 0.15) if year % 3 == 0 else random.uniform(-0.05, 0.08)
            year_return += (base_return + volatility) * (pct / 100)
        
        annual_returns.append(year_return)
    
    # Calculate metrics
    total_return = sum(annual_returns)
    avg_return = total_return / years
    max_drawdown = min(annual_returns)
    volatility = sum(abs(r - avg_return) for r in annual_returns) / years
    sharpe_ratio = (avg_return - 0.02) / volatility if volatility > 0 else 0
    
    return {
        "annual_returns": annual_returns,
        "total_return": total_return,
        "average_return": avg_return,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "years": years
    }


def compare_backtest_strategies(allocation1: dict, allocation2: dict, label1: str, label2: str, years: int = 10) -> str:
    """Compare two portfolio strategies with backtesting"""
    
    backtest1 = backtest_portfolio(allocation1, years)
    backtest2 = backtest_portfolio(allocation2, years)
    
    prompt = f"""Compare these two backtested portfolio strategies:

**{label1}:**
- Allocation: {allocation1}
- Total Return ({years} years): {backtest1['total_return']:.1%}
- Average Annual Return: {backtest1['average_return']:.1%}
- Volatility: {backtest1['volatility']:.1%}
- Max Drawdown: {backtest1['max_drawdown']:.1%}
- Sharpe Ratio: {backtest1['sharpe_ratio']:.2f}

**{label2}:**
- Allocation: {allocation2}
- Total Return ({years} years): {backtest2['total_return']:.1%}
- Average Annual Return: {backtest2['average_return']:.1%}
- Volatility: {backtest2['volatility']:.1%}
- Max Drawdown: {backtest2['max_drawdown']:.1%}
- Sharpe Ratio: {backtest2['sharpe_ratio']:.2f}

Provide:
1. Which strategy performed better and why
2. Risk/return trade-offs
3. Which strategy suits different investor types

Keep under 250 words.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to compare strategies: {str(e)}"


def backtest_crisis_period(allocation: dict, crisis: str) -> str:
    """Show how portfolio would have performed during specific crisis"""
    
    crisis_impacts = {
        "2008 financial crisis": {
            "stocks": -0.37,
            "bonds": 0.05,
            "gold": 0.05,
            "real estate": -0.35,
            "cash": 0.02
        },
        "covid-19 crash": {
            "stocks": -0.34,  # Initial drop
            "bonds": 0.08,
            "gold": 0.25,
            "real estate": -0.10,
            "cash": 0.01
        },
        "dot-com bubble": {
            "stocks": -0.49,
            "bonds": 0.11,
            "gold": -0.03,
            "real estate": 0.05,
            "cash": 0.04
        }
    }
    
    crisis_lower = crisis.lower()
    impact_data = None
    
    for key, data in crisis_impacts.items():
        if key in crisis_lower or crisis_lower in key:
            impact_data = data
            break
    
    if not impact_data:
        return f"No historical data available for '{crisis}'. Try: 2008 financial crisis, COVID-19 crash, or dot-com bubble."
    
    # Calculate portfolio impact
    portfolio_impact = 0
    for asset, pct in allocation.items():
        asset_lower = asset.lower()
        if "stock" in asset_lower or "s&p" in asset_lower:
            portfolio_impact += impact_data["stocks"] * (pct / 100)
        elif "bond" in asset_lower:
            portfolio_impact += impact_data["bonds"] * (pct / 100)
        elif "gold" in asset_lower:
            portfolio_impact += impact_data["gold"] * (pct / 100)
        elif "real estate" in asset_lower or "reit" in asset_lower:
            portfolio_impact += impact_data["real estate"] * (pct / 100)
        elif "cash" in asset_lower:
            portfolio_impact += impact_data["cash"] * (pct / 100)
    
    prompt = f"""Analyze how this portfolio would have performed during the {crisis}:

Portfolio: {allocation}
Estimated Impact: {portfolio_impact:.1%}

Historical context:
- Stocks: {impact_data['stocks']:.1%}
- Bonds: {impact_data['bonds']:.1%}
- Gold: {impact_data['gold']:.1%}

Explain:
1. What happened to this portfolio during the crisis
2. Which assets protected value and which lost value
3. Key lessons for future crisis preparation

Keep under 200 words.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to analyze crisis performance: {str(e)}"

# Create portfolios directory if it doesn't exist
PORTFOLIOS_DIR = "saved_portfolios"
if not os.path.exists(PORTFOLIOS_DIR):
    os.makedirs(PORTFOLIOS_DIR)


def save_portfolio_snapshot(name: str) -> dict:
    """
    Save current portfolio and session state to a snapshot.
    Returns the saved snapshot data.
    """
    snapshot = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "profile": {
            "age": st.session_state.get("age", 25),
            "risk_tolerance": st.session_state.get("risk_tolerance", "Moderate"),
            "savings": st.session_state.get("savings", 10000),
            "has_401k": st.session_state.get("has_401k", False),
            "retirement_balance": st.session_state.get("retirement_balance", 0),
            "has_stocks": st.session_state.get("has_stocks", False),
            "stock_value": st.session_state.get("stock_value", 0),
            "experience": st.session_state.get("experience", "Beginner"),
            "time_available": st.session_state.get("time_available", "3-5 hours"),
        },
        "portfolio": {
            "stocks": st.session_state.get("stocks", 0.0),
            "bonds": st.session_state.get("bonds", 0.0),
            "cash": st.session_state.get("cash", 0.0),
            "real_estate": st.session_state.get("real_estate", 0.0),
            "crypto": st.session_state.get("crypto", 0.0),
            "other": st.session_state.get("other", 0.0),
        },
        "goal_state": st.session_state.get("goal_state"),
        "glossary": st.session_state.get("glossary", []),
    }
    
    # Save to file
    filename = f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(PORTFOLIOS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    return snapshot


def load_all_saved_portfolios() -> list:
    """Load all saved portfolios from disk"""
    portfolios = []
    
    if not os.path.exists(PORTFOLIOS_DIR):
        return portfolios
    
    for filename in os.listdir(PORTFOLIOS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(PORTFOLIOS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    snapshot = json.load(f)
                    snapshot['filename'] = filename  # Store filename for deletion
                    portfolios.append(snapshot)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Sort by timestamp (newest first)
    portfolios.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return portfolios


def delete_portfolio_snapshot(filename: str) -> None:
    """Delete a saved portfolio file"""
    filepath = os.path.join(PORTFOLIOS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)


def load_portfolio_snapshot(snapshot: dict) -> None:
    """
    Load a portfolio snapshot into current session state.
    Must be called BEFORE widgets are created.
    """
    # Store in a temp variable that will be applied on next rerun
    st.session_state["_pending_load"] = snapshot

def apply_pending_load():
    """Apply pending portfolio load if exists"""
    if "_pending_load" not in st.session_state:
        return
    
    snapshot = st.session_state["_pending_load"]
    
    # Load profile
    profile = snapshot.get("profile", {})
    st.session_state["age"] = profile.get("age", 25)
    st.session_state["risk_tolerance"] = profile.get("risk_tolerance", "Moderate")
    st.session_state["savings"] = profile.get("savings", 10000)
    st.session_state["has_401k"] = profile.get("has_401k", False)
    st.session_state["retirement_balance"] = profile.get("retirement_balance", 0)
    st.session_state["has_stocks"] = profile.get("has_stocks", False)
    st.session_state["stock_value"] = profile.get("stock_value", 0)
    st.session_state["experience"] = profile.get("experience", "Beginner")
    st.session_state["time_available"] = profile.get("time_available", "3-5 hours")
    
    # Load portfolio allocation
    portfolio = snapshot.get("portfolio", {})
    st.session_state["stocks"] = portfolio.get("stocks", 0.0)
    st.session_state["bonds"] = portfolio.get("bonds", 0.0)
    st.session_state["cash"] = portfolio.get("cash", 0.0)
    st.session_state["real_estate"] = portfolio.get("real_estate", 0.0)
    st.session_state["crypto"] = portfolio.get("crypto", 0.0)
    st.session_state["other"] = portfolio.get("other", 0.0)
    
    # Load goal state and glossary
    st.session_state["goal_state"] = snapshot.get("goal_state")
    st.session_state["glossary"] = snapshot.get("glossary", [])
    
    # Clear pending load and set message
    del st.session_state["_pending_load"]
    st.session_state.load_message = f"✅ Loaded '{snapshot['name']}'"

def serialize_snapshot(snapshot: dict) -> str:
    """Serialize snapshot to base64 string for download"""
    # Remove filename key if present (not needed for export)
    export_snapshot = {k: v for k, v in snapshot.items() if k != 'filename'}
    json_str = json.dumps(export_snapshot, indent=2)
    return base64.b64encode(json_str.encode()).decode()


def deserialize_snapshot(serialized: str) -> dict:
    """Deserialize snapshot from base64 string"""
    json_str = base64.b64decode(serialized.encode()).decode()
    return json.loads(json_str)

# ============================
# User Story 7 – Reset Memory
# ============================

def reset_all_memory():
    """
    Clears every key from session_state.
    Required by tests:
    - After calling reset_all_memory(), NO keys must remain.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def reset_profile_and_memory():
    """
    Fully reset:
    - Clear ALL session memory
    - Restore DEFAULT profile values
    - DO NOT add goal_state or glossary back (tests require they are absent)
    """
    # Full wipe
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # === Restore Default Profile Values ===
    st.session_state["age"] = 25
    st.session_state["risk_tolerance"] = "Moderate"
    st.session_state["savings"] = 10000
    st.session_state["has_401k"] = False
    st.session_state["retirement_balance"] = 0
    st.session_state["has_stocks"] = False
    st.session_state["stock_value"] = 0
    st.session_state["experience"] = "Beginner"
    st.session_state["time_available"] = "3-5 hours"


# ============= LangGraph Types & Nodes =============


class GoalState(TypedDict):
    goal: str
    subtasks: List[dict]
    completed_tasks: List[str]
    user_info: dict
    messages: List[str]
    portfolio: Dict[str, float]
    portfolio_analysis: str


def decompose_goal(state: GoalState) -> GoalState:
    """Decompose user goal into subtasks using OpenAI"""
    goal = state["goal"]
    user_info = state.get("user_info", {})

    prompt = f"""You are a financial education mentor. Break down this financial learning goal into 4-5 concrete, actionable subtasks.

Goal: {goal}

User Financial Situation:
- Age: {user_info.get('age', 'not specified')}
- Risk Tolerance: {user_info.get('risk_tolerance', 'not specified')}
- Current Savings: ${user_info.get('savings', 0):,}
- Has 401(k): {user_info.get('has_401k', False)} (Balance: ${user_info.get('retirement_balance', 0):,})
- Owns Stocks: {user_info.get('has_stocks', False)} (Value: ${user_info.get('stock_value', 0):,})

User Experience Level: {user_info.get('experience', 'beginner')}
Available Time per Week: {user_info.get('time', 'not specified')}

IMPORTANT - Risk Tolerance Filtering:
- Risk Tolerance: {user_info.get('risk_tolerance', 'Moderate')}
- If Conservative: Focus learning on bonds, dividend stocks, and index funds. Mention but de-emphasize high-risk options.
- If Moderate: Include balanced mix of growth and income investments.
- If Aggressive: Include growth stocks, sector-specific investments, and higher-volatility assets in learning path.

Create subtasks that are:
- Specific and measurable
- Ordered from beginner to advanced
- Realistic and achievable
- Focused on financial literacy
- Tailored to their financial situation

Return ONLY a numbered list of subtasks, nothing else.
Example format:
1. Learn the basics of personal budgeting
2. Understand different types of bank accounts
3. Research investment options
...
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        response_text = response.choices[0].message.content

        lines = [ln for ln in response_text.strip().splitlines() if ln.strip()]
        subtasks: List[dict] = []
        for idx, ln in enumerate(lines, start=1):
            parts = re.split(r"^\s*\d+[\.\)]\s*", ln, maxsplit=1)
            task_text = parts[-1].strip() if len(parts) > 1 else ln.strip()
            if task_text:
                subtasks.append({"id": idx, "task": task_text, "completed": False})

        state["subtasks"] = subtasks
        state["messages"].append(f"Generated {len(subtasks)} subtasks")

    except Exception as e:
        st.error(f"Error generating subtasks: {str(e)}")
        state["messages"].append(f"Error: {str(e)}")

    return state


def analyze_portfolio(state: GoalState) -> GoalState:
    """Analyze user's portfolio and provide guidance"""
    portfolio = state.get("portfolio", {})
    user_info = state.get("user_info", {})
    if not portfolio:
        return state

    portfolio_desc = ", ".join([f"{k}: {v}%" for k, v in portfolio.items()])
    total = sum(portfolio.values())

    prompt = f"""You are a financial advisor analyzing a client's investment portfolio.

Client Profile:
- Experience Level: {user_info.get('experience', 'beginner')}
- Portfolio: {portfolio_desc}
- Total Allocation: {total}%

Provide a comprehensive portfolio analysis including:

1. **Portfolio Health Assessment**: Rate the portfolio (Excellent/Good/Needs Improvement/Poor)
2. **Diversification Analysis**: Comment on the current asset allocation
3. **Risk Assessment**: Identify any concentration risks or concerns
4. **Specific Recommendations**: Provide 3-5 actionable recommendations to improve the portfolio
5. **Explanation**: Explain WHY each recommendation matters for their financial health

Be specific, educational, and actionable. Tailor advice to their experience level.
Format your response in clear sections with headers.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        state["portfolio_analysis"] = response.choices[0].message.content
        state["messages"].append("Portfolio analysis complete")
    except Exception as e:
        st.error(f"Error analyzing portfolio: {str(e)}")
        state["messages"].append(f"Portfolio analysis error: {str(e)}")
    return state


def display_tasks(state: GoalState) -> GoalState:
    state["messages"].append("Tasks ready for display")
    return state


def create_goal_mentor_graph():
    workflow = StateGraph(GoalState)
    workflow.add_node("decompose", decompose_goal)
    workflow.add_node("analyze_portfolio", analyze_portfolio)
    workflow.add_node("display", display_tasks)

    workflow.add_edge("decompose", "analyze_portfolio")
    workflow.add_edge("analyze_portfolio", "display")
    workflow.add_edge("display", END)

    workflow.set_entry_point("decompose")
    return workflow.compile()


graph = create_goal_mentor_graph()

# ============= Streamlit App =============


def main():
    st.set_page_config(
        page_title="Finance Learning Bot - AI Goal Mentor",
        page_icon="💰",
        layout="wide",
    )
    init_session_state()
    apply_pending_load()

    st.title("💰 Finance Learning Bot - AI Goal Mentor")

    # Display load/save messages
    if st.session_state.get("load_message"):
        st.info(st.session_state.load_message)
        st.session_state.load_message = ""

    # -------- Sidebar: Profile + Portfolio + Glossary --------
    with st.sidebar:
        st.header("📋 Your Profile")
        
        # ===== CORE PROFILE (Always Visible) =====
        age = st.number_input("Age", min_value=18, max_value=100, value=25, key="age")
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1, key="risk_tolerance"
        )
        
        risk_colors = {
            "Conservative": "🟢",
            "Moderate": "🟡",
            "Aggressive": "🔴",
        }
        
        risk_descriptions = {
            "Conservative": "Capital preservation, steady income",
            "Moderate": "Balanced growth with moderate risk",
            "Aggressive": "Maximum growth, higher volatility",
        }
        
        st.caption(
            f"{risk_colors.get(risk_tolerance, '🟡')} {risk_descriptions.get(risk_tolerance, '')}"
        )
        
        # ===== ADDITIONAL PROFILE DETAILS (Expandable) =====
        with st.expander("💼 Additional Profile Details", expanded=False):
            experience = st.selectbox(
                "Financial Knowledge", ["Beginner", "Intermediate", "Advanced"], key="experience"
            )
            time_available = st.selectbox(
                "Time per Week", ["1-2 hours", "3-5 hours", "5-10 hours", "10+ hours"], key="time_available"
            )
            savings = st.number_input(
                "Current Savings ($)", min_value=0, value=10000, step=1000, key="savings"
            )
            has_401k = st.checkbox("I have a 401(k)", value=False, key="has_401k")
            retirement_balance = (
                st.number_input(
                    "401(k) Balance ($)", min_value=0, value=0, step=1000, key="retirement_balance"
                )
                if has_401k
                else 0
            )
            has_stocks = st.checkbox("I own stocks", value=False, key="has_stocks")
            stock_value = (
                st.number_input(
                    "Stock Portfolio Value ($)", min_value=0, value=0, step=1000, key="stock_value"
                )
                if has_stocks
                else 0
            )
        
        st.markdown("---")
        
        # ===== INVESTMENT TOOLS (Expandable) =====
        with st.expander("💡 Investment Tools", expanded=False):
            st.markdown("**Get Recommendations**")
            if st.button(
                "📋 Get Risk-Based Recommendations",
                type="secondary",
                use_container_width=True,
                help=f"Get investment options for {risk_tolerance} risk",
            ):
                with st.spinner("Generating recommendations..."):
                    recommendations = get_investment_recommendations(risk_tolerance, "")
                    st.session_state["investment_recommendations"] = {
                        "risk_tolerance": risk_tolerance,
                        "recommendations": recommendations,
                    }
                st.success("✅ Recommendations generated!")
                st.rerun()
            
            st.markdown("**Portfolio Allocation**")
            if st.button(
                "🎯 Suggest Allocation",
                type="secondary",
                use_container_width=True,
                help=f"Get {risk_tolerance} allocation suggestion",
            ):
                with st.spinner(f"Generating {risk_tolerance} allocation..."):
                    allocation = suggest_portfolio_allocation(risk_tolerance)
                    st.session_state["suggested_allocation"] = {
                        "risk_tolerance": risk_tolerance,
                        "allocation": allocation,
                        "timestamp": "current",
                    }
                    st.session_state.allocation_history.append(
                        {
                            "label": f"{risk_tolerance} Allocation",
                            "allocation": allocation,
                        }
                    )
                st.success("✅ Allocation generated!")
                st.rerun()
            
            # User feedback for adjustment
            if st.session_state.get("suggested_allocation"):
                st.markdown("**Adjust Allocation**")
                user_adjustment = st.text_input(
                    "Feedback",
                    placeholder="e.g., 'I want more stocks'",
                    key="user_adjustment_input",
                )
                
                if st.button("🔧 Adjust", use_container_width=True):
                    if user_adjustment.strip():
                        current_alloc = st.session_state.suggested_allocation["allocation"]
                        with st.spinner("Adjusting allocation..."):
                            new_alloc, explanation = adjust_allocation_by_feedback(
                                current_alloc, user_adjustment
                            )
                            st.session_state.suggested_allocation = {
                                "risk_tolerance": risk_tolerance,
                                "allocation": new_alloc,
                                "adjustment_explanation": explanation,
                                "timestamp": "adjusted",
                            }
                        st.success("✅ Adjusted!")
                        st.rerun()
        
        # ===== PORTFOLIO ANALYSIS (Expandable) =====
        with st.expander("📊 Portfolio Analysis", expanded=False):
            st.caption("Enter your current asset allocation")
            
            stocks = st.number_input(
                "Stocks", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="stocks"
            )
            bonds = st.number_input(
                "Bonds", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="bonds"
            )
            cash = st.number_input(
                "Cash/Savings", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="cash"
            )
            real_estate = st.number_input(
                "Real Estate", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="real_estate"
            )
            crypto = st.number_input(
                "Cryptocurrency", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="crypto"
            )
            other = st.number_input(
                "Other", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="other"
            )
            
            total_allocation = stocks + bonds + cash + real_estate + crypto + other
            if total_allocation == 100:
                st.success(f"✅ Total: {total_allocation}%")
            elif total_allocation > 0:
                st.warning(f"⚠️ Total: {total_allocation}% (need 100%)")
            else:
                st.info("💡 Enter your portfolio allocation")
            
            analyze_enabled = total_allocation == 100.0
            if st.button(
                "🔍 Analyze Portfolio",
                type="primary",
                disabled=not analyze_enabled,
                help="Total must equal 100% to analyze" if not analyze_enabled else "Analyze your portfolio",
                use_container_width=True,
            ):
                portfolio: Dict[str, float] = {}
                if stocks > 0:
                    portfolio["Stocks"] = stocks
                if bonds > 0:
                    portfolio["Bonds"] = bonds
                if cash > 0:
                    portfolio["Cash"] = cash
                if real_estate > 0:
                    portfolio["Real Estate"] = real_estate
                if crypto > 0:
                    portfolio["Cryptocurrency"] = crypto
                if other > 0:
                    portfolio["Other"] = other
                
                with st.spinner("Analyzing your portfolio..."):
                    portfolio_state: GoalState = {
                        "goal": "",
                        "subtasks": [],
                        "completed_tasks": [],
                        "user_info": {
                            "experience": experience.lower(),
                            "time": time_available,
                        },
                        "messages": [],
                        "portfolio": portfolio,
                        "portfolio_analysis": "",
                    }
                    analyzed_state = analyze_portfolio(portfolio_state)
                    st.session_state.portfolio_analysis_only = {
                        "portfolio": portfolio,
                        "analysis": analyzed_state.get("portfolio_analysis", ""),
                    }
                st.success("✅ Analysis complete!")
                st.rerun()
        
        # ===== SAVE/LOAD PORTFOLIO (Expandable) =====
        with st.expander("💾 Save/Load Portfolio", expanded=False):
            st.markdown("**Save Current Session**")
            snapshot_name = st.text_input(
                "Portfolio Name",
                placeholder="e.g., 'My Conservative Portfolio'",
                key="snapshot_name_input"
            )
            
            if st.button("💾 Save Portfolio", use_container_width=True):
                if snapshot_name.strip():
                    with st.spinner("Saving portfolio..."):
                        snapshot = save_portfolio_snapshot(snapshot_name.strip())
                    st.success(f"✅ Saved '{snapshot_name}'!")
                    st.rerun()
                else:
                    st.warning("Please enter a name")
            
            # Load Portfolio Section
            saved_portfolios = load_all_saved_portfolios()
            
            if saved_portfolios:
                st.markdown("**Load Saved Portfolio**")
                
                portfolio_options = [
                    f"{p['name']} - {datetime.fromisoformat(p['timestamp']).strftime('%Y-%m-%d %H:%M')}"
                    for p in saved_portfolios
                ]
                
                selected_idx = st.selectbox(
                    "Select Portfolio",
                    options=range(len(portfolio_options)),
                    format_func=lambda x: portfolio_options[x],
                    key="portfolio_select"
                )
                
                col_load, col_delete = st.columns(2)
                
                with col_load:
                    if st.button("📂 Load", use_container_width=True):
                        snapshot = saved_portfolios[selected_idx]
                        load_portfolio_snapshot(snapshot)
                        st.rerun()
                
                with col_delete:
                    if st.button("🗑️ Delete", use_container_width=True):
                        snapshot_to_delete = saved_portfolios[selected_idx]
                        deleted_name = snapshot_to_delete['name']
                        delete_portfolio_snapshot(snapshot_to_delete['filename'])
                        st.session_state.load_message = f"🗑️ Deleted '{deleted_name}'"
                        st.rerun()
                
                st.markdown("**Export/Import**")
                
                snapshot_to_export = saved_portfolios[selected_idx]
                serialized = serialize_snapshot(snapshot_to_export)
                
                st.download_button(
                    label="⬇️ Export",
                    data=serialized,
                    file_name=f"{snapshot_to_export['name'].replace(' ', '_')}.portfolio",
                    mime="text/plain",
                    use_container_width=True
                )
                
                uploaded_file = st.file_uploader(
                    "Import Portfolio File",
                    type=["portfolio"],
                    key="portfolio_import"
                )
                
                if uploaded_file is not None:
                    try:
                        serialized_data = uploaded_file.read().decode()
                        imported_snapshot = deserialize_snapshot(serialized_data)
                        imported_name = imported_snapshot.get('name', 'Imported Portfolio')
                        imported_snapshot['timestamp'] = datetime.now().isoformat()
                        filename = f"{imported_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        filepath = os.path.join(PORTFOLIOS_DIR, filename)
                        with open(filepath, 'w') as f:
                            json.dump(imported_snapshot, f, indent=2)
                        st.success(f"✅ Imported '{imported_name}'!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to import: {str(e)}")
            else:
                st.info("No saved portfolios yet")
        
        # ===== RESET OPTIONS (Expandable) =====
        with st.expander("🧹 Reset Options", expanded=False):
            if st.button("🧹 Reset Memory", use_container_width=True):
                reset_all_memory()
                st.success("Memory cleared!")
                st.rerun()
            
            if st.button("🔄 Start New Session", use_container_width=True):
                reset_profile_and_memory()
                st.success("New session started!")
                st.rerun()

        # ===== GLOSSARY (Only show if exists) =====
        if st.session_state.glossary:
            st.markdown("---")
            st.subheader("📘 My Glossary")
            for term, definition in st.session_state.glossary:
                with st.expander(term):
                    render_explanation_card(definition, term=term)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                if st.button("🧹 Clear Glossary"):
                    st.session_state.glossary = []
                    st.session_state.explanation = ""
                    st.session_state.reasoning_answer = ""
                    st.rerun()
            with col_g2:
                export = "\n\n".join(
                    [f"{i+1}. {t}\n{d}" for i, (t, d) in enumerate(st.session_state.glossary)]
                )
                st.download_button(
                    "⬇️ Download",
                    data=export.encode("utf-8"),
                    file_name="glossary.txt",
                    mime="text/plain",
                )

    # -------- Main Content --------

    # Display Investment Recommendations
    if (
        "investment_recommendations" in st.session_state
        and st.session_state.investment_recommendations
    ):
        rec_data = st.session_state.investment_recommendations

        st.subheader(f"💡 Investment Recommendations ({rec_data['risk_tolerance']} Risk)")
        st.markdown(rec_data["recommendations"])

        if st.button("🗑️ Clear Recommendations"):
            st.session_state.investment_recommendations = None
            st.rerun()

        st.markdown("---")

    # Display Suggested Allocation (User Story 5)
    if (
        "suggested_allocation" in st.session_state
        and st.session_state.suggested_allocation
    ):
        alloc_data = st.session_state.suggested_allocation

        st.subheader(
            f"🎯 Suggested Portfolio Allocation ({alloc_data['risk_tolerance']} Risk)"
        )

        allocation = alloc_data["allocation"]
        st.markdown("**Recommended Allocation:**")
        cols = st.columns(len(allocation))
        for i, (asset, pct) in enumerate(allocation.items()):
            with cols[i]:
                st.metric(asset, f"{pct}%")

        total = sum(allocation.values())
        if total != 100:
            st.warning(f"⚠️ Total: {total}% (should be 100%)")
        else:
            st.success(f"✅ Total: {total}%")

        if "adjustment_explanation" in alloc_data:
            st.markdown("---")
            st.markdown("**💡 Adjustment Explanation:**")
            st.info(alloc_data["adjustment_explanation"])

        col_compare1, col_compare2 = st.columns(2)

        with col_compare1:
            compare_risk = st.selectbox(
                "Compare with:",
                ["Conservative", "Moderate", "Aggressive"],
                key="compare_risk_select",
            )

        with col_compare2:
            if st.button("🔍 Compare", use_container_width=True):
                if compare_risk != alloc_data["risk_tolerance"]:
                    with st.spinner(f"Generating {compare_risk} allocation..."):
                        compare_alloc = suggest_portfolio_allocation(compare_risk)
                        comparison_text = compare_portfolios(
                            allocation,
                            compare_alloc,
                            alloc_data["risk_tolerance"],
                            compare_risk,
                        )

                        st.session_state.portfolio_comparison = {
                            "current": {
                                "label": alloc_data["risk_tolerance"],
                                "allocation": allocation,
                            },
                            "alternative": {
                                "label": compare_risk,
                                "allocation": compare_alloc,
                            },
                            "comparison": comparison_text,
                        }
                    st.rerun()

        if st.session_state.get("portfolio_comparison"):
            comp_data = st.session_state.portfolio_comparison

            st.markdown("---")
            st.markdown("### 📊 Portfolio Comparison")

            col_port1, col_port2 = st.columns(2)

            with col_port1:
                st.markdown(f"**{comp_data['current']['label']}**")
                for asset, pct in comp_data["current"]["allocation"].items():
                    st.write(f"{asset}: {pct}%")

            with col_port2:
                st.markdown(f"**{comp_data['alternative']['label']}**")
                for asset, pct in comp_data["alternative"]["allocation"].items():
                    st.write(f"{asset}: {pct}%")

            st.markdown("---")
            st.markdown("**Analysis:**")
            st.markdown(comp_data["comparison"])

            if st.button("🗑️ Clear Comparison"):
                st.session_state.portfolio_comparison = None
                st.rerun()

        if st.button("🗑️ Clear Allocation"):
            st.session_state.suggested_allocation = None
            st.session_state.portfolio_comparison = None
            st.rerun()

        st.markdown("---")

    # Portfolio-only Analysis
    if st.session_state.portfolio_analysis_only:
        st.subheader("💼 Your Portfolio Analysis")
        pdata = st.session_state.portfolio_analysis_only

        st.markdown("**Current Allocation:**")
        portfolio = pdata.get("portfolio", {})
        if portfolio:
            cols = st.columns(len(portfolio))
            for i, (asset, pct) in enumerate(portfolio.items()):
                with cols[i]:
                    st.metric(asset, f"{pct}%")

        st.markdown("---")
        analysis = pdata.get("analysis", "")
        if analysis:
            st.markdown(analysis)

        if st.button("🗑️ Clear Analysis"):
            st.session_state.portfolio_analysis_only = None
            st.rerun()

        st.markdown("---")

    # Goal + Progress
    col1, col2 = st.columns([2, 1])

    with col1:
        # ============= UNIFIED MENTOR INTERFACE =============
        action_type = st.radio(
            "What would you like to do?",
            options=["📚 Generate Learning Path", "❓ Explain a Concept", "🤔 Explain Why"],
            horizontal=True,
            key="mentor_action_type",
            help="📚 Generate Learning Path: Create a step-by-step plan | ❓ Explain a Concept: Get simple explanations | 🤔 Explain Why: Understand the reasoning"
        )
        if action_type == "📚 Generate Learning Path":
            placeholder = "Example: Learn about stock market investing, Understand retirement planning, Master personal budgeting..."
            help_text = "Enter your financial learning goal"
            label_text = "Enter your financial learning goal"
        elif action_type == "❓ Explain a Concept":
            placeholder = "Example: What is a dividend yield? What are bonds?"
            help_text = "Ask about any financial term or concept"
            label_text = "Ask about any financial term or concept"
        else:  # "🤔 Explain Why"
            placeholder = 'Examples: "Why 30% in bonds?", "Why not cryptocurrency?", "Why learn about bonds first?"'
            help_text = "Ask why the bot recommended something"
            label_text = "Ask why the bot recommended something"
        
        with st.form(key="mentor_form", clear_on_submit=False):
            user_input = st.text_input(
                label_text,
                placeholder=placeholder,
                key="unified_mentor_input",
                help=help_text
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Submit", type="primary", use_container_width=True)
        
        if submitted:
            if user_input.strip():
                if action_type == "📚 Generate Learning Path":
                    # Generate Learning Path logic
                    with st.spinner("Creating your personalized learning path..."):
                        portfolio_for_goal: Dict[str, float] = {}
                        if stocks > 0:
                            portfolio_for_goal["Stocks"] = stocks
                        if bonds > 0:
                            portfolio_for_goal["Bonds"] = bonds
                        if cash > 0:
                            portfolio_for_goal["Cash"] = cash
                        if real_estate > 0:
                            portfolio_for_goal["Real Estate"] = real_estate
                        if crypto > 0:
                            portfolio_for_goal["Cryptocurrency"] = crypto
                        if other > 0:
                            portfolio_for_goal["Other"] = other

                        initial_state: GoalState = {
                            "goal": user_input,
                            "subtasks": [],
                            "completed_tasks": [],
                            "user_info": {
                                "age": age,
                                "risk_tolerance": risk_tolerance,
                                "savings": savings,
                                "has_401k": has_401k,
                                "retirement_balance": retirement_balance,
                                "has_stocks": has_stocks,
                                "stock_value": stock_value,
                                "experience": experience.lower(),
                                "time": time_available,
                            },
                            "messages": [],
                            "portfolio": portfolio_for_goal,
                            "portfolio_analysis": "",
                        }
                        result = graph.invoke(initial_state)
                        st.session_state.goal_state = result
                    st.success("✅ Learning path created successfully!")
                    st.rerun()
                    
                elif action_type == "❓ Explain a Concept":
                    # Explain Concept logic
                    with st.spinner("Teaching you finance wisdom..."):
                        answer = explain_finance_term(user_input)
                        st.session_state.explanation = answer
                        st.session_state.glossary.append((user_input, answer))
                    st.success("✅ Explained!")
                    render_explanation_card(answer, user_input)
                    
                else:  # "🤔 Explain Why"
                    # Explain Why logic
                    with st.spinner("Explaining the reasoning..."):
                        reasoning = explain_recommendation(
                            user_input,
                            risk_tolerance=risk_tolerance,
                            age=age,
                            time_available=time_available,
                            experience=experience,
                            suggested_allocation=st.session_state.get("suggested_allocation"),
                            investment_recs=st.session_state.get("investment_recommendations"),
                            goal_state=st.session_state.get("goal_state"),
                        )
                        st.session_state.reasoning_answer = reasoning
                        st.session_state.glossary.append((f"Why: {user_input}", reasoning))
                    st.success("✅ Reasoning explained!")
                    render_explanation_card(reasoning, term="Why this recommendation?")
            else:
                st.warning("Please enter your question or goal first!")
        
        if not submitted:
            if st.session_state.get("explanation") and action_type == "❓ Explain a Concept":
                # Get the last glossary entry if it exists
                if st.session_state.glossary:
                    last_term, last_def = st.session_state.glossary[-1]
                    if last_def == st.session_state.explanation:
                        render_explanation_card(last_def, last_term)
                
            if st.session_state.get("reasoning_answer") and action_type == "🤔 Explain Why":
                render_explanation_card(st.session_state.reasoning_answer, term="Why this recommendation?")

    with col2:
        st.subheader("📊 Progress Overview")
        gs = st.session_state.get("goal_state")
        if gs and gs.get("subtasks"):
            total_tasks = len(gs["subtasks"])
            completed = sum(1 for t in gs["subtasks"] if t["completed"])
            progress = (completed / total_tasks * 100) if total_tasks > 0 else 0.0
            
            # Display metrics side-by-side
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Total Tasks", total_tasks)
            with col_metric2:
                st.metric("Completed", completed)
            
            st.progress(progress / 100)
            st.caption(f"{progress:.0f}% Complete")
        else:
            st.info("Enter a goal to see your progress!")

    # Display Learning Path
    gs = st.session_state.get("goal_state")
    if gs and gs.get("subtasks"):
        st.markdown("---")
        st.subheader("📚 Your Learning Path")
        st.markdown(f"**Goal:** {gs['goal']}")
        st.markdown("### Tasks")

        for idx, task in enumerate(gs["subtasks"]):
            col_task, col_button = st.columns([4, 1])

            with col_task:
                if task["completed"]:
                    st.markdown(f"~~{idx + 1}. {task['task']}~~ ✅")
                else:
                    st.markdown(f"{idx + 1}. **{task['task']}**")

            with col_button:
                if not task["completed"]:
                    if st.button("✓ Complete", key=f"complete_{idx}"):
                        st.session_state.goal_state["subtasks"][idx]["completed"] = True
                        st.rerun()

        st.markdown("---")
        col_reset, col_export = st.columns(2)
        with col_reset:
            if st.button("🔄 Start New Goal"):
                st.session_state.goal_state = None
                st.rerun()
        with col_export:
            if st.button("🔥 Export Learning Path"):
                export_text = f"Finance Learning Goal: {gs['goal']}\n\n"

                if gs.get("portfolio"):
                    export_text += "Portfolio Allocation:\n"
                    for asset, pct in gs["portfolio"].items():
                        export_text += f"  {asset}: {pct}%\n"
                    export_text += "\n"

                if gs.get("portfolio_analysis"):
                    export_text += "Portfolio Analysis:\n"
                    export_text += gs["portfolio_analysis"] + "\n\n"

                export_text += "Learning Path:\n"
                for i, t in enumerate(gs["subtasks"]):
                    status = "✅" if t["completed"] else "⬜"
                    export_text += f"{i + 1}. {status} {t['task']}\n"

                st.download_button(
                    label="Download as Text",
                    data=export_text,
                    file_name="learning_path.txt",
                    mime="text/plain",
                )

    st.markdown("---")
    st.subheader("📊 Historical Performance Data")
    
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        st.markdown("**Asset Class Performance**")
        asset_query = st.selectbox(
            "Select Asset Class:",
            ["Index Funds", "S&P 500", "Bonds", "Stocks", "Real Estate", "Other"],
            key="asset_performance_select"
        )

        # Show text input if "Other" is selected
        if asset_query == "Other":
            asset_query = st.text_input(
                "Specify Asset Class:",
                placeholder="e.g., Commodities, International Stocks, Treasury Bills",
                key="custom_asset_input"
            )

            if not asset_query or not asset_query.strip():
                st.warning("⚠️ Please enter an asset class")

        if st.button("📈 View Historical Performance", use_container_width=True):
            if asset_query and asset_query.strip() and asset_query != "Other":
                with st.spinner(f"Retrieving data for {asset_query}..."):
                    hist_data = get_historical_performance(asset_query)
                    st.session_state.historical_data_display = {
                        'type': 'asset_performance',
                        'data': hist_data,
                        'asset': asset_query
                    }
                st.rerun()
    
    with col_hist2:
        st.markdown("**Market Events Analysis**")
        event_query = st.selectbox(
            "Select Market Event:",
            ["2008 Financial Crisis", "COVID-19 Crash", "Dot-com Bubble", "Other"],
            key="market_event_select"
        )

        # Show text input if "Other" is selected
        if event_query == "Other":
            event_query = st.text_input(
                "Specify Market Event:",
                placeholder="e.g., Black Monday 1987, Asian Financial Crisis, Crypto Winter",
                key="custom_event_input"
            )

            if not event_query or not event_query.strip():
                st.warning("⚠️ Please enter a market event")

        if st.button("📉 View Event Impact", use_container_width=True):
            if event_query and event_query.strip() and event_query != "Other":
                with st.spinner(f"Analyzing {event_query}..."):
                    event_data = get_market_event_data(event_query)
                    st.session_state.historical_data_display = {
                        'type': 'market_event',
                        'data': event_data,
                        'event': event_query
                    }
                st.rerun()
    
    # Display historical data if exists
    if st.session_state.get('historical_data_display'):
        hist_display = st.session_state.historical_data_display
        st.markdown("---")
        
        if hist_display['type'] == 'asset_performance':
            st.markdown(f"### 📊 {hist_display['asset']} - Historical Performance")
            st.markdown(hist_display['data'])
        elif hist_display['type'] == 'market_event':
            st.markdown(f"### 📉 {hist_display['event']} - Impact Analysis")
            st.markdown(hist_display['data'])
        elif hist_display['type'] == 'allocation_comparison':
            st.markdown("### 📊 Portfolio Allocation Comparison")
            st.markdown(hist_display['data'])
        
        if st.button("🗑️ Clear Historical Data"):
            st.session_state.historical_data_display = None
            st.rerun()
        
        st.markdown("---")
    
    # Portfolio comparison with historical evidence
    st.markdown("**Compare Portfolio Allocations (Historical Evidence)**")
    
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    
    with col_comp1:
        alloc1 = st.text_input(
            "Portfolio 1:",
            placeholder="e.g., 60% stocks / 40% bonds",
            key="hist_alloc1"
        )
    
    with col_comp2:
        alloc2 = st.text_input(
            "Portfolio 2:",
            placeholder="e.g., 100% stocks",
            key="hist_alloc2"
        )
    
    with col_comp3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("⚖️ Compare", use_container_width=True, key="compare_historical_allocations"):
            if alloc1.strip() and alloc2.strip():
                with st.spinner("Comparing historical performance..."):
                    comparison = compare_allocation_performance(alloc1, alloc2)
                    st.session_state.backtest_results = None
                    st.session_state.strategy_comparison = None
                    st.session_state.crisis_backtest = None
                    st.session_state.historical_data_display = {
                        'type': 'allocation_comparison',
                        'data': comparison,
                        'alloc1': alloc1,
                        'alloc2': alloc2
                    }
                st.rerun()
            else:
                st.warning("Please enter both portfolio allocations to compare")

    # ============= User Story 9: Backtesting UI =============
    st.markdown("---")
    st.subheader("🔬 Portfolio Backtesting")
    st.markdown("*Test how your portfolio would have performed historically*")
    
    tab1, tab2, tab3 = st.tabs(["📈 Backtest Portfolio", "⚖️ Compare Strategies", "💥 Crisis Performance"])
    
    with tab1:
        st.markdown("**Test Your Portfolio Allocation**")
        
        # Use current allocation from sidebar if available
        current_portfolio = {}
        if stocks > 0:
            current_portfolio["Stocks"] = stocks
        if bonds > 0:
            current_portfolio["Bonds"] = bonds
        if cash > 0:
            current_portfolio["Cash"] = cash
        if real_estate > 0:
            current_portfolio["Real Estate"] = real_estate
        if crypto > 0:
            current_portfolio["Cryptocurrency"] = crypto
        if other > 0:
            current_portfolio["Other"] = other
        
        col_bt1, col_bt2 = st.columns([3, 1])
        
        with col_bt1:
            if current_portfolio:
                portfolio_str = ", ".join([f"{k}: {v}%" for k, v in current_portfolio.items()])
                st.info(f"Current allocation: {portfolio_str}")
            else:
                st.warning("Enter portfolio allocation in sidebar first")
        
        with col_bt2:
            backtest_years = st.selectbox("Years:", [5, 10, 20, 30], index=1, key="backtest_years")
        
        if st.button("🔬 Run Backtest", disabled=not current_portfolio, use_container_width=True):
            with st.spinner(f"Backtesting over {backtest_years} years..."):
                results = backtest_portfolio(current_portfolio, backtest_years)
                st.session_state.backtest_results = {
                    "allocation": current_portfolio,
                    "results": results
                }
            st.success("✅ Backtest complete!")
            st.rerun()
    
    with tab2:
        st.markdown("**Compare Two Investment Strategies**")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("**Strategy 1**")
            s1_stocks = st.number_input("Stocks %", 0.0, 100.0, 80.0, 5.0, key="s1_stocks")
            s1_bonds = st.number_input("Bonds %", 0.0, 100.0, 20.0, 5.0, key="s1_bonds")
            s1_other = 100 - s1_stocks - s1_bonds
            st.caption(f"Other: {s1_other}%")
        
        with col_s2:
            st.markdown("**Strategy 2**")
            s2_stocks = st.number_input("Stocks %", 0.0, 100.0, 60.0, 5.0, key="s2_stocks")
            s2_bonds = st.number_input("Bonds %", 0.0, 100.0, 40.0, 5.0, key="s2_bonds")
            s2_other = 100 - s2_stocks - s2_bonds
            st.caption(f"Other: {s2_other}%")
        
        compare_years = st.selectbox("Years:", [5, 10, 20, 30], index=1, key="compare_years")
        
        if st.button("⚖️ Compare Strategies", use_container_width=True):
            strategy1 = {"Stocks": s1_stocks, "Bonds": s1_bonds}
            if s1_other > 0:
                strategy1["Other"] = s1_other
            
            strategy2 = {"Stocks": s2_stocks, "Bonds": s2_bonds}
            if s2_other > 0:
                strategy2["Other"] = s2_other
            
            with st.spinner("Comparing strategies..."):
                comparison = compare_backtest_strategies(
                    strategy1, strategy2,
                    f"{int(s1_stocks)}/{int(s1_bonds)} Strategy",
                    f"{int(s2_stocks)}/{int(s2_bonds)} Strategy",
                    compare_years
                )
                
                # Also get individual backtests for metrics
                bt1 = backtest_portfolio(strategy1, compare_years)
                bt2 = backtest_portfolio(strategy2, compare_years)
                
                st.session_state.strategy_comparison = {
                    "strategy1": {"allocation": strategy1, "backtest": bt1, "label": f"{int(s1_stocks)}/{int(s1_bonds)}"},
                    "strategy2": {"allocation": strategy2, "backtest": bt2, "label": f"{int(s2_stocks)}/{int(s2_bonds)}"},
                    "comparison": comparison,
                    "years": compare_years
                }
            st.success("✅ Comparison complete!")
            st.rerun()
    
    with tab3:
        st.markdown("**How Would Your Portfolio Have Performed During a Crisis?**")
        
        if current_portfolio:
            portfolio_str = ", ".join([f"{k}: {v}%" for k, v in current_portfolio.items()])
            st.info(f"Testing allocation: {portfolio_str}")
        else:
            st.warning("Enter portfolio allocation in sidebar first")
        
        crisis_event = st.selectbox(
            "Select Crisis Event:",
            ["2008 Financial Crisis", "COVID-19 Crash", "Dot-com Bubble"],
            key="crisis_select"
        )
        
        if st.button("💥 Test Crisis Performance", disabled=not current_portfolio, use_container_width=True):
            with st.spinner(f"Analyzing {crisis_event} performance..."):
                crisis_result = backtest_crisis_period(current_portfolio, crisis_event)
                st.session_state.crisis_backtest = {
                    "allocation": current_portfolio,
                    "crisis": crisis_event,
                    "analysis": crisis_result
                }
            st.success("✅ Crisis analysis complete!")
            st.rerun()
    
    # Display backtest results
    if st.session_state.get("backtest_results"):
        st.markdown("---")
        st.markdown("### 📊 Backtest Results")
        
        bt_data = st.session_state.backtest_results
        results = bt_data["results"]
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Total Return", f"{results['total_return']:.1%}", 
                     help=f"Total return over {results['years']} years")
        with col_m2:
            st.metric("Avg Annual Return", f"{results['average_return']:.1%}",
                     help="Average return per year")
        with col_m3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}",
                     help="Largest single-year loss", delta_color="inverse")
        with col_m4:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}",
                     help="Risk-adjusted return (higher is better)")
        
        st.markdown("**Year-by-Year Returns:**")
        years_data = [(f"Year {i+1}", f"{ret:.1%}") for i, ret in enumerate(results["annual_returns"])]
        cols = st.columns(min(len(years_data), 5))
        for i, (year, ret) in enumerate(years_data):
            with cols[i % 5]:
                st.caption(f"{year}: {ret}")
        
        if st.button("🗑️ Clear Backtest"):
            st.session_state.backtest_results = None
            st.rerun()
        
        st.markdown("---")
    
    # Display strategy comparison
    if st.session_state.get("strategy_comparison"):
        st.markdown("---")
        st.markdown("### ⚖️ Strategy Comparison Results")
        
        comp_data = st.session_state.strategy_comparison
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown(f"**Strategy 1: {comp_data['strategy1']['label']}**")
            bt1 = comp_data['strategy1']['backtest']
            st.metric("Total Return", f"{bt1['total_return']:.1%}")
            st.metric("Volatility", f"{bt1['volatility']:.1%}")
            st.metric("Sharpe Ratio", f"{bt1['sharpe_ratio']:.2f}")
        
        with col_c2:
            st.markdown(f"**Strategy 2: {comp_data['strategy2']['label']}**")
            bt2 = comp_data['strategy2']['backtest']
            st.metric("Total Return", f"{bt2['total_return']:.1%}")
            st.metric("Volatility", f"{bt2['volatility']:.1%}")
            st.metric("Sharpe Ratio", f"{bt2['sharpe_ratio']:.2f}")
        
        st.markdown("---")
        st.markdown("**Analysis:**")
        st.markdown(comp_data['comparison'])
        
        if st.button("🗑️ Clear Comparison"):
            st.session_state.strategy_comparison = None
            st.rerun()
        
        st.markdown("---")
    
    # Display crisis backtest
    if st.session_state.get("crisis_backtest"):
        st.markdown("---")
        st.markdown(f"### 💥 {st.session_state.crisis_backtest['crisis']} - Portfolio Performance")
        
        crisis_data = st.session_state.crisis_backtest
        st.markdown(crisis_data['analysis'])
        
        if st.button("🗑️ Clear Crisis Analysis"):
            st.session_state.crisis_backtest = None
            st.rerun()
        
        st.markdown("---")


if __name__ == "__main__":
    main()