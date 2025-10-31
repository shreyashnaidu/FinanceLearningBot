# Finance Learning Bot - AI Goal Mentor

An AI-powered investment education assistant built with LangGraph and Streamlit.

## Team Members
- Abhishek Harish Thumar
- Shreyash Naidu Mamidi
- Kyle Lobo

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

## Running Tests

```bash
python -m unittest test_user_story_1.py -v
```

## Git Commit

To link your commit to Issue #1:

```bash
git add app.py test_user_story_1.py README.md
git commit -m "Implement User Story 1: Financial situation inputs - Fixes #1"
git push origin main
```

The keyword `Fixes #1` will automatically close the issue on GitHub.