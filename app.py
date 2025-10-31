import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt.tool_node import ToolNode
# env config
load_dotenv()
st.set_page_config(page_title="Hotel QA Agent", page_icon="🏨", layout="centered")

@st.cache_data(show_spinner=False)
def load_hotel_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find dataset at '{csv_path}'. Place hotels.csv here or set HOTELS_CSV."
        )
    df = pd.read_csv(csv_path)

    required = [
        "hotel_id", "hotel_name", "city", "country",
        "lat", "lon", "star_rating",
        "cleanliness_base", "comfort_base", "facilities_base",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["city"] = df["city"].astype(str).str.strip().str.lower()
    df["country"] = df["country"].astype(str).str.strip().str.lower()

    num_cols = ["star_rating", "cleanliness_base", "comfort_base", "facilities_base"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].fillna(0)

    df["hotel_name"] = df["hotel_name"].astype(str).str.strip()
    return df

HOTELS_CSV = os.getenv("HOTELS_CSV", "hotels.csv")
hotels_df = load_hotel_data(HOTELS_CSV)

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Return a simple GitHub-style markdown table for a DataFrame (no tabulate)."""
    if df.empty:
        return ""
    headers = list(df.columns)
    # Header row
    lines = ["|" + "|".join(headers) + "|"]
    # Separator row
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    # Data rows
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.1f}")
            else:
                cells.append(str(val))
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)

@tool
def query_hotels(
    city: str = None,
    country: str = None,
    min_star: float = None,
    min_cleanliness: float = None,
    min_comfort: float = None,
    min_facilities: float = None,
    sort_by: str = None,
    limit: int = 5
) -> str:
    """
    Search the hotel dataset by filters and return a markdown table.
    """
    df = hotels_df.copy()

    if city:
        city = str(city).strip().lower()
        df = df[df["city"].str.contains(city, na=False)]
    if country:
        country = str(country).strip().lower()
        df = df[df["country"].str.contains(country, na=False)]
    if min_star is not None:
        df = df[df["star_rating"] >= float(min_star)]
    if min_cleanliness is not None:
        df = df[df["cleanliness_base"] >= float(min_cleanliness)]
    if min_comfort is not None:
        df = df[df["comfort_base"] >= float(min_comfort)]
    if min_facilities is not None:
        df = df[df["facilities_base"] >= float(min_facilities)]

    if df.empty:
        return (
            "No hotels matched these filters. "
            "Try broadening your search (e.g., lower minimums or check spelling for city/country)."
        )

    try:
        limit = int(limit) if limit is not None else 5
    except Exception:
        limit = 5
    limit = max(1, min(limit, 10))

    valid_sort = ["star_rating", "cleanliness_base", "comfort_base", "facilities_base"]
    if sort_by in valid_sort:
        df = df.sort_values(by=sort_by, ascending=False)

    cols = [
        "hotel_name", "city", "country",
        "star_rating", "cleanliness_base", "comfort_base", "facilities_base"
    ]
    df = df[cols].head(limit).copy()
    df.columns = ["Hotel Name", "City", "Country", "Star Rating", "Cleanliness", "Comfort", "Facilities"]

    return dataframe_to_markdown(df)

SYSTEM_PROMPT = (
    "You are a helpful travel assistant. "
    "When the user asks about hotels or data from the hotel catalog, "
    "use the `query_hotels` tool to retrieve results.\n\n"
    "Instructions:\n"
    "1) If a user asks for hotels by city/country or by rating thresholds, call the tool with appropriate arguments.\n"
    "2) In your final answer, include a brief helpful sentence and the returned table.\n"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Set it in your environment or .env file before running.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools([query_hotels])

def llm_node(state: MessagesState) -> MessagesState:
    """Call the LLM (bound with tools); may return tool calls or final text."""
    messages: list[AnyMessage] = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def route_after_llm(state: MessagesState):
    """If the LLM asked for a tool, go to tool node; else end."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", llm_node)
graph.add_node("tools", ToolNode([query_hotels]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_after_llm, ["tools", END])
graph.add_edge("tools", "agent")
agent = graph.compile()

st.title("🏨 Hotel QA Agent")
st.caption("Ask me for hotels by city/country and rating thresholds. I’ll query the dataset and show results.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# Render previous messages (skip system)
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(msg.content)
        else:
            st.write(msg.content)

# Chat input
user_text = st.chat_input("e.g., Top 5 hotels in Paris by cleanliness")
if user_text:
    with st.chat_message("user"):
        st.write(user_text)
    st.session_state.messages.append(HumanMessage(content=user_text))

    state = {"messages": st.session_state.messages}
    final = agent.invoke(state)

    last = final["messages"][-1]
    st.session_state.messages.append(last)
    with st.chat_message("assistant"):
        st.markdown(last.content)
