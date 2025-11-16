import os
import re
import json
from typing import TypedDict, List, Dict
from html import escape

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
    ss.setdefault("reasoning_answer", "")        # NEW: User Story 6
    ss.setdefault("historical_data_display", None) 


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


def explain_recommendation(user_question: str, last_recommendation: str, user_info: dict) -> str:
    """
    User Story 6: Explain *why* a recommendation was made.
    Required signature for automated tests.
    """
    age = user_info.get("age", "not provided")
    risk = user_info.get("risk_tolerance", "not provided")
    time_available = user_info.get("time_available", "not provided")
    experience = user_info.get("experience", "not provided")

    # Build user background text
    profile_text = f"""
    - Age: {age}
    - Risk Tolerance: {risk}
    - Experience Level: {experience}
    - Time Available per Week: {time_available}
    """

    prompt = f"""
    A user is asking WHY the following financial recommendation was made:

    Recommendation:
    "{last_recommendation}"

    User question:
    "{user_question}"

    User Profile:
    {profile_text}

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

    st.title("💰 Finance Learning Bot - AI Goal Mentor")
    st.markdown("### Your Personal Financial Education Assistant")

    # -------- Sidebar: Profile + Portfolio + Glossary --------
    with st.sidebar:
        st.header("📋 Your Profile")

        experience = st.selectbox(
            "Financial Knowledge Level", ["Beginner", "Intermediate", "Advanced"]
        )
        time_available = st.selectbox(
            "Time Available per Week", ["1-2 hours", "3-5 hours", "5-10 hours", "10+ hours"]
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        savings = st.number_input(
            "Current Savings ($)", min_value=0, value=10000, step=1000
        )
        has_401k = st.checkbox("I have a 401(k)", value=False)
        retirement_balance = (
            st.number_input(
                "401(k) Balance ($)", min_value=0, value=0, step=1000
            )
            if has_401k
            else 0
        )
        has_stocks = st.checkbox("I own stocks", value=False)
        stock_value = (
            st.number_input(
                "Stock Portfolio Value ($)", min_value=0, value=0, step=1000
            )
            if has_stocks
            else 0
        )

        risk_tolerance = st.selectbox(
            "Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1
        )

        risk_colors = {
            "Conservative": "🟢",
            "Moderate": "🟡",
            "Aggressive": "🔴",
        }

        risk_descriptions = {
            "Conservative": "Focus: Capital preservation, steady income",
            "Moderate": "Focus: Balanced growth with moderate risk",
            "Aggressive": "Focus: Maximum growth, higher volatility",
        }

        st.info(
            f"{risk_colors.get(risk_tolerance, '🟡')} **{risk_tolerance}**: {risk_descriptions.get(risk_tolerance, '')}"
        )

        st.markdown("---")
        

        # Investment Recommendations Section
        st.header("💡 Investment Recommendations")

        if st.button(
            "📋 Get Risk-Based Recommendations",
            type="secondary",
            use_container_width=True,
            help=f"Get investment options suitable for {risk_tolerance} risk tolerance",
        ):
            with st.spinner("Generating recommendations..."):
                recommendations = get_investment_recommendations(risk_tolerance, "")

                st.session_state["investment_recommendations"] = {
                    "risk_tolerance": risk_tolerance,
                    "recommendations": recommendations,
                }

            st.success("✅ Recommendations generated!")
            st.rerun()

        st.markdown("---")

        # Portfolio Allocation Suggestions (User Story 5)
        st.header("📈 Portfolio Allocation")

        if st.button(
            "🎯 Suggest Allocation",
            type="secondary",
            use_container_width=True,
            help=f"Get {risk_tolerance} portfolio allocation suggestion",
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
        st.markdown("---")
        st.subheader("🧹 Reset Options")
        # Reset memory only
        if st.button("🧹 Reset Memory", use_container_width=True):
            reset_all_memory()
            st.success("Memory cleared! Your recommendations and history have been reset.")
            st.rerun()
        # Full session reset
        if st.button("🔄 Start New Session", use_container_width=True):
            reset_profile_and_memory()
            st.success("New session started! All profile data cleared.")
            st.rerun()


        # User feedback for adjustment (User Story 5)
        if st.session_state.get("suggested_allocation"):
            st.markdown("**Adjust Allocation:**")
            user_adjustment = st.text_input(
                "Feedback",
                placeholder="e.g., 'I want more stocks' or 'Less bonds'",
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

        st.markdown("---")
        st.header("📊 Portfolio Analysis")
        st.markdown("*Get personalized portfolio guidance*")

        st.markdown("**Current Asset Allocation (%)**")
        stocks = st.number_input(
            "Stocks", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="stocks"
        )
        bonds = st.number_input(
            "Bonds", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="bonds"
        )
        cash = st.number_input(
            "Cash/Savings",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            key="cash",
        )
        real_estate = st.number_input(
            "Real Estate",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            key="real_estate",
        )
        crypto = st.number_input(
            "Cryptocurrency",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            key="crypto",
        )
        other = st.number_input(
            "Other", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="other"
        )

        total_allocation = stocks + bonds + cash + real_estate + crypto + other
        if total_allocation == 100:
            st.success(f"✅ Total Allocation: {total_allocation}%")
        elif total_allocation > 0:
            st.warning(
                f"⚠️ Total Allocation: {total_allocation}% (Must be 100%)"
            )
        else:
            st.info("💡 Total Allocation: 0% (Enter your portfolio)")

        analyze_enabled = total_allocation == 100.0
        if st.button(
            "🔍 Analyze Portfolio",
            type="primary",
            disabled=not analyze_enabled,
            help=(
                "Total allocation must equal 100% to analyze"
                if not analyze_enabled
                else "Click to analyze your portfolio"
            ),
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

            st.success("✅ Portfolio analysis complete!")
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This AI mentor breaks down learning goals and provides portfolio guidance."
        )

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
        st.subheader("🎯 Enter Your Financial Learning Goal")
        goal_input = st.text_area(
            "What would you like to learn about finance?",
            placeholder=(
                "Example: Learn about stock market investing, Understand retirement "
                "planning, Master personal budgeting..."
            ),
            height=100,
        )

        if st.button("🚀 Generate Learning Path", type="primary"):
            if goal_input.strip():
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
                        "goal": goal_input,
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
            else:
                st.warning("Please enter a goal first!")

    with col2:
        st.subheader("📊 Progress Overview")
        gs = st.session_state.get("goal_state")
        if gs and gs.get("subtasks"):
            total_tasks = len(gs["subtasks"])
            completed = sum(1 for t in gs["subtasks"] if t["completed"])
            progress = (completed / total_tasks * 100) if total_tasks > 0 else 0.0
            st.metric("Total Tasks", total_tasks)
            st.metric("Completed", completed)
            st.progress(progress / 100)
            st.caption(f"{progress:.0f}% Complete")
        else:
            st.info("Enter a goal to see your progress!")

    # Ask Your Mentor (concept explanations)
    st.markdown("---")
    st.subheader("💬 Ask Your Mentor")
    user_question = st.text_input(
        "Confused about a financial term?",
        placeholder="Example: What is a dividend yield?",
    )
    if st.button("❓ Explain This"):
        if user_question.strip():
            with st.spinner("Teaching you finance wisdom..."):
                answer = explain_finance_term(user_question)
                st.session_state.explanation = answer
                st.session_state.glossary.append((user_question, answer))
            st.success("✅ Explained!")
        else:
            st.warning("Please enter a question.")
    if st.session_state.explanation:
        render_explanation_card(st.session_state.explanation, user_question)

    # NEW: User Story 6 – "Why?" about recommendations
    st.markdown("---")
    st.subheader("🤔 Ask 'Why?' About My Plan")
    why_q = st.text_input(
        "Ask why the bot recommended something (allocations, avoiding assets, or the learning order).",
        placeholder=(
            'Examples: "Why 30% in bonds and not 20%?", '
            '"Why not cryptocurrency?", '
            '"Why learn about bonds before stocks?"'
        ),
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
        
        if st.button("📈 View Historical Performance", use_container_width=True):
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
        
        if st.button("📉 View Event Impact", use_container_width=True):
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
        if st.button("⚖️ Compare", use_container_width=True):
            if alloc1.strip() and alloc2.strip():
                with st.spinner("Comparing historical performance..."):
                    comparison = compare_allocation_performance(alloc1, alloc2)
                    st.session_state.historical_data_display = {
                        'type': 'allocation_comparison',
                        'data': comparison,
                        'alloc1': alloc1,
                        'alloc2': alloc2
                    }
                st.rerun()
            else:
                st.warning("Please enter both portfolio allocations to compare")

    st.markdown("---")

    if st.button("🧠 Explain the reasoning"):
        if why_q.strip():
            with st.spinner("Explaining the reasoning behind your recommendations..."):
                reasoning = explain_recommendation(
                    why_q,
                    risk_tolerance=risk_tolerance,
                    age=age,
                    time_available=time_available,
                    experience=experience,
                    suggested_allocation=st.session_state.get("suggested_allocation"),
                    investment_recs=st.session_state.get("investment_recommendations"),
                    goal_state=st.session_state.get("goal_state"),
                )
                st.session_state.reasoning_answer = reasoning
                # Save to glossary so user can review later
                st.session_state.glossary.append((f"Why: {why_q}", reasoning))
            st.success("✅ Reasoning explained!")
        else:
            st.warning("Please type a 'why' question first.")

    if st.session_state.get("reasoning_answer"):
        render_explanation_card(
            st.session_state.reasoning_answer,
            term="Why this recommendation?",
        )

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
            if st.button("📥 Export Learning Path"):
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


if __name__ == "__main__":
    main()
