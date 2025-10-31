import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key
if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found! Please set your OPENAI_API_KEY in the .env file")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"❌ Error initializing OpenAI: {str(e)}")
    st.stop()

# Define State for LangGraph
class GoalState(TypedDict):
    goal: str
    subtasks: List[dict]
    completed_tasks: List[str]
    user_info: dict
    messages: List[str]

# LangGraph Node Functions
def decompose_goal(state: GoalState) -> GoalState:
    """Decompose user goal into subtasks using OpenAI"""
    goal = state["goal"]
    user_info = state.get("user_info", {})
    
    prompt = f"""You are a financial education mentor. Break down this financial learning goal into 5-7 concrete, actionable subtasks.

Goal: {goal}

User Experience Level: {user_info.get('experience', 'beginner')}
Available Time per Week: {user_info.get('time', 'not specified')}

Create subtasks that are:
- Specific and measurable
- Ordered from beginner to advanced
- Realistic and achievable
- Focused on financial literacy

Return ONLY a numbered list of subtasks, nothing else.
Example format:
1. Learn the basics of personal budgeting
2. Understand different types of bank accounts
3. Research investment options
...
"""
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        # Parse the response into structured subtasks
        subtasks_text = response_text.strip().split('\n')
        subtasks = []
        for idx, task in enumerate(subtasks_text):
            if task.strip() and any(char.isdigit() for char in task[:3]):
                # Remove numbering and clean up
                task_text = task.split('.', 1)[-1].strip()
                if task_text:
                    subtasks.append({
                        'id': idx,
                        'task': task_text,
                        'completed': False
                    })
        
        state["subtasks"] = subtasks
        state["messages"].append(f"Generated {len(subtasks)} subtasks")
        
    except Exception as e:
        st.error(f"Error generating subtasks: {str(e)}")
        state["messages"].append(f"Error: {str(e)}")
    
    return state

def display_tasks(state: GoalState) -> GoalState:
    """Mark tasks as ready for display"""
    state["messages"].append("Tasks ready for display")
    return state

# Build LangGraph
def create_goal_mentor_graph():
    workflow = StateGraph(GoalState)
    
    # Add nodes
    workflow.add_node("decompose", decompose_goal)
    workflow.add_node("display", display_tasks)
    
    # Add edges
    workflow.add_edge("decompose", "display")
    workflow.add_edge("display", END)
    
    # Set entry point
    workflow.set_entry_point("decompose")
    
    return workflow.compile()

# Initialize the graph
graph = create_goal_mentor_graph()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Finance Learning Bot - AI Goal Mentor",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Finance Learning Bot - AI Goal Mentor")
    st.markdown("### Your Personal Financial Education Assistant")
    
    # Initialize session state
    if 'goal_state' not in st.session_state:
        st.session_state.goal_state = None
    
    # Sidebar for user information
    with st.sidebar:
        st.header("📋 Your Profile")
        experience = st.selectbox(
            "Financial Knowledge Level",
            ["Beginner", "Intermediate", "Advanced"]
        )
        time_available = st.selectbox(
            "Time Available per Week",
            ["1-2 hours", "3-5 hours", "5-10 hours", "10+ hours"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This AI mentor helps you break down financial learning goals into actionable steps.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Enter Your Financial Learning Goal")
        
        # Goal input
        goal_input = st.text_area(
            "What would you like to learn about finance?",
            placeholder="Example: Learn about stock market investing, Understand retirement planning, Master personal budgeting...",
            height=100
        )
        
        if st.button("🚀 Generate Learning Path", type="primary"):
            if goal_input.strip():
                with st.spinner("Creating your personalized learning path..."):
                    # Create initial state
                    initial_state = {
                        "goal": goal_input,
                        "subtasks": [],
                        "completed_tasks": [],
                        "user_info": {
                            "experience": experience.lower(),
                            "time": time_available
                        },
                        "messages": []
                    }
                    
                    # Run the graph
                    result = graph.invoke(initial_state)
                    st.session_state.goal_state = result
                    
                st.success("✅ Learning path created successfully!")
            else:
                st.warning("Please enter a goal first!")
    
    with col2:
        st.subheader("📊 Progress Overview")
        if st.session_state.goal_state and st.session_state.goal_state.get('subtasks'):
            total_tasks = len(st.session_state.goal_state['subtasks'])
            completed = sum(1 for task in st.session_state.goal_state['subtasks'] if task['completed'])
            progress = (completed / total_tasks * 100) if total_tasks > 0 else 0
            
            st.metric("Total Tasks", total_tasks)
            st.metric("Completed", completed)
            st.progress(progress / 100)
            st.caption(f"{progress:.0f}% Complete")
        else:
            st.info("Enter a goal to see your progress!")
    
    # Display subtasks
    if st.session_state.goal_state and st.session_state.goal_state.get('subtasks'):
        st.markdown("---")
        st.subheader("📚 Your Learning Path")
        st.markdown(f"**Goal:** {st.session_state.goal_state['goal']}")
        
        st.markdown("### Tasks")
        
        for idx, task in enumerate(st.session_state.goal_state['subtasks']):
            col_task, col_button = st.columns([4, 1])
            
            with col_task:
                if task['completed']:
                    st.markdown(f"~~{idx + 1}. {task['task']}~~ ✅")
                else:
                    st.markdown(f"{idx + 1}. **{task['task']}**")
            
            with col_button:
                if not task['completed']:
                    if st.button("✓ Complete", key=f"complete_{idx}"):
                        st.session_state.goal_state['subtasks'][idx]['completed'] = True
                        st.rerun()
        
        # Reset button
        st.markdown("---")
        col_reset, col_export = st.columns(2)
        
        with col_reset:
            if st.button("🔄 Start New Goal"):
                st.session_state.goal_state = None
                st.rerun()
        
        with col_export:
            if st.button("📥 Export Learning Path"):
                # Create text export
                export_text = f"Finance Learning Goal: {st.session_state.goal_state['goal']}\n\n"
                export_text += "Learning Path:\n"
                for idx, task in enumerate(st.session_state.goal_state['subtasks']):
                    status = "✅" if task['completed'] else "⬜"
                    export_text += f"{idx + 1}. {status} {task['task']}\n"
                
                st.download_button(
                    label="Download as Text",
                    data=export_text,
                    file_name="learning_path.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()