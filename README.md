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

