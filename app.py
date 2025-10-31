import os
import re
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
        unsafe_allow_html=True
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
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# ============= LangGraph Types & Nodes =============

class GoalState(TypedDict):
    goal: str
    subtasks: List[dict]
    completed_tasks: List[str]
    user_info: dict
    messages: List[str]
    portfolio: Dict[str, float]        # NEW: Portfolio allocation
    portfolio_analysis: str            # NEW: Portfolio analysis result

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
            max_tokens=500
        )
        response_text = response.choices[0].message.content

        # Parse lines "1. Task" / "1) Task" safely
        lines = [ln for ln in response_text.strip().splitlines() if ln.strip()]
        subtasks = []
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
            max_tokens=800
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
    st.set_page_config(page_title="Finance Learning Bot - AI Goal Mentor", page_icon="💰", layout="wide")
    init_session_state()

    st.title("💰 Finance Learning Bot - AI Goal Mentor")
    st.markdown("### Your Personal Financial Education Assistant")

    # -------- Sidebar: Profile + Portfolio + Glossary --------
    with st.sidebar:
        st.header("📋 Your Profile")

        # Financial Situation inputs
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
        savings = st.number_input("Current Savings ($)", min_value=0, value=10000, step=1000)
        has_401k = st.checkbox("I have a 401(k)", value=False)
        retirement_balance = st.number_input("401(k) Balance ($)", min_value=0, value=0, step=1000) if has_401k else 0
        has_stocks = st.checkbox("I own stocks", value=False)
        stock_value = st.number_input("Stock Portfolio Value ($)", min_value=0, value=0, step=1000) if has_stocks else 0

        st.markdown("---")
        experience = st.selectbox("Financial Knowledge Level", ["Beginner", "Intermediate", "Advanced"])
        time_available = st.selectbox("Time Available per Week", ["1-2 hours", "3-5 hours", "5-10 hours", "10+ hours"])

        st.markdown("---")
        st.header("📊 Portfolio Analysis")
        st.markdown("*Get personalized portfolio guidance*")

        st.markdown("**Current Asset Allocation (%)**")
        stocks = st.number_input("Stocks", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="stocks")
        bonds = st.number_input("Bonds", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="bonds")
        cash = st.number_input("Cash/Savings", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="cash")
        real_estate = st.number_input("Real Estate", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="real_estate")
        crypto = st.number_input("Cryptocurrency", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="crypto")
        other = st.number_input("Other", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="other")

        total_allocation = stocks + bonds + cash + real_estate + crypto + other
        if total_allocation == 100:
            st.success(f"✅ Total Allocation: {total_allocation}%")
        elif total_allocation > 0:
            st.warning(f"⚠️ Total Allocation: {total_allocation}% (Must be 100%)")
        else:
            st.info("💡 Total Allocation: 0% (Enter your portfolio)")

        analyze_enabled = (total_allocation == 100.0)
        if st.button(
            "🔍 Analyze Portfolio",
            type="primary",
            disabled=not analyze_enabled,
            help="Total allocation must equal 100% to analyze" if not analyze_enabled else "Click to analyze your portfolio",
            use_container_width=True
        ):
            portfolio = {}
            if stocks > 0: portfolio["Stocks"] = stocks
            if bonds > 0: portfolio["Bonds"] = bonds
            if cash > 0: portfolio["Cash"] = cash
            if real_estate > 0: portfolio["Real Estate"] = real_estate
            if crypto > 0: portfolio["Cryptocurrency"] = crypto
            if other > 0: portfolio["Other"] = other

            with st.spinner("Analyzing your portfolio..."):
                portfolio_state = {
                    "goal": "",
                    "subtasks": [],
                    "completed_tasks": [],
                    "user_info": {"experience": experience.lower(), "time": time_available},
                    "messages": [],
                    "portfolio": portfolio,
                    "portfolio_analysis": ""
                }
                analyzed_state = analyze_portfolio(portfolio_state)

                st.session_state.portfolio_analysis_only = {
                    "portfolio": portfolio,
                    "analysis": analyzed_state.get("portfolio_analysis", "")
                }

            st.success("✅ Portfolio analysis complete!")
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.info("This AI mentor breaks down learning goals and provides portfolio guidance.")

        # Glossary panel
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
                    st.rerun()
            with col_g2:
                export = "\n\n".join([f"{i+1}. {t}\n{d}" for i, (t, d) in enumerate(st.session_state.glossary)])
                st.download_button(
                    "⬇️ Download",
                    data=export.encode("utf-8"),
                    file_name="glossary.txt",
                    mime="text/plain"
                )

    # -------- Portfolio-only Analysis (center) --------
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

    # -------- Goal + Progress --------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Enter Your Financial Learning Goal")
        goal_input = st.text_area(
            "What would you like to learn about finance?",
            placeholder="Example: Learn about stock market investing, Understand retirement planning, Master personal budgeting...",
            height=100
        )

        if st.button("🚀 Generate Learning Path", type="primary"):
            if goal_input.strip():
                with st.spinner("Creating your personalized learning path..."):
                    portfolio = {}
                    if stocks > 0: portfolio["Stocks"] = stocks
                    if bonds > 0: portfolio["Bonds"] = bonds
                    if cash > 0: portfolio["Cash"] = cash
                    if real_estate > 0: portfolio["Real Estate"] = real_estate
                    if crypto > 0: portfolio["Cryptocurrency"] = crypto
                    if other > 0: portfolio["Other"] = other

                    initial_state = {
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
                            "time": time_available
                        },
                        "messages": [],
                        "portfolio": portfolio,
                        "portfolio_analysis": ""
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
            progress = (completed / total_tasks * 100) if total_tasks > 0 else 0
            st.metric("Total Tasks", total_tasks)
            st.metric("Completed", completed)
            st.progress(progress / 100)
            st.caption(f"{progress:.0f}% Complete")
        else:
            st.info("Enter a goal to see your progress!")

    # -------- Ask Your Mentor (Concept Explainer) --------
    st.markdown("---")
    st.subheader("💬 Ask Your Mentor")
    user_question = st.text_input(
        "Confused about a financial term?",
        placeholder="Example: What is a dividend yield?"
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

    # -------- Display Learning Path --------
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
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()

