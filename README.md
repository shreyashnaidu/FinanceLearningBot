# Finance Learning Bot - AI Goal Mentor

An AI-powered investment education assistant built with LangGraph and Streamlit.

## Team Members
- Abhishek Harish Thumar
- Shreyash Naidu Mamidi
- Kyle Lobo

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```
## First Iteration

### User Story 1: Financial Situation Input and Goal Decomposition
**Issue:** [#1](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/1)

**Implementation:**
- Added financial situation inputs (age, risk tolerance, savings, 401k, stocks)
- AI generates 4-5 personalized learning phases based on user's financial situation
- All 3 scenarios implemented and tested

**Changes Made:**
1. `decompose_goal()` - Updated prompt to include financial situation context
2. Sidebar - Added financial input fields (age, risk tolerance, savings, 401k, stocks)
3. `initial_state` - Now includes financial information in user_info

**Test Results:**
```bash
python -m unittest test_user_story_1.py -v
```
- ✅ Scenario 1: Generates 4-5 learning phases
- ✅ Scenario 2: Tailors to age and risk tolerance
- ✅ Scenario 3: Acknowledges current assets

## Running Tests

```bash
python -m unittest test_user_story_1.py -v
```

### User Story 2: Portfolio Guidance and Health Assessment
**Issue:** [#2](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/2)

**Implementation:**
- Added portfolio input form with 6 asset classes (stocks, bonds, cash, real estate, crypto, other)
- "Analyze Portfolio" button enabled only when total allocation = 100%
- AI analyzes portfolio composition and provides personalized investment guidance
- Identifies risks (concentration, lack of diversification) based on experience level
- All 3 scenarios implemented and tested

**Changes Made:**
1. `analyze_portfolio()` - New LangGraph node for portfolio health assessment
2. `GoalState` - Added `portfolio: Dict[str, float]` and `portfolio_analysis: str` fields
3. Sidebar - Added portfolio input section with real-time total calculation
4. Main content - Displays portfolio analysis with clear/export functionality
5. LangGraph workflow - Updated to include portfolio analysis node

**Test Results:**
```bash
pytest tests/test_user_story_2.py -v
```
- ✅ Scenario 1: Unbalanced portfolio (60% stocks, 40% cash) - suggests bond allocation
- ✅ Scenario 2: Beginner with no investments (100% cash) - recommends index funds
- ✅ Scenario 3: High risk portfolio (100% tech stocks) - identifies concentration risk

**UI Features:**
- Real-time portfolio total validation with color-coded feedback
- Disabled button state when total ≠ 100%
- Standalone portfolio analysis (no goal required)
- Combined portfolio + learning goal analysis
- Export functionality includes portfolio data


## Running Tests

```bash
# User Story 2: Portfolio Guidance
pytest tests/test_user_story_2.py -v
```

### User Story 3: Progress Tracking & Learning Flow
**Issue:** [#3](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/3)

**Implementation:**
- Real-time learning progress tracking
- Task completion checkmarks + strike-through for finished tasks
- Automatically highlights next recommended task
- Phase-based unlocking (Phase 2 only visible once Phase 1 completed)
- Session state preserves task progress
- Added “Ask Mentor” glossary-based learning support (stores explained terms)


**Changes Made:**
1. Updated GoalState to store task completion status
2. Added progress metrics and completion bar in UI
3. Added task button logic to mark as complete and auto-advance
4. Added glossary sidebar to store terms + explanations
5. Integrated “Ask Mentor” section to explain finance terms in-app

**Test Results:**
```bash
python -m unittest tests/test_user_story_3.py -v
```
- ✅ Scenario 1: Progress bar increases correctly as tasks are completed (33% at 5/15 tasks)
- ✅ Scenario 2: Completing a task activates the next task in sequence
- ✅ Scenario 3: Phase 2 tasks remain locked until all Phase 1 tasks are completed
- ✅ Scenario 3: Phase 2 tasks remain locked until all Phase 1 tasks are completed

**UI Features:**
- Strike-through with ✅ indicator for completed tasks
- Automatic “next task” activation to guide learning flow
- Sequential phase unlocking to maintain structured learning
- Glossary panel to review previously explained financial terms
- “Explain This” assistant input for real-time concept clarification


## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v

```
