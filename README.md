# Finance Learning Bot - AI Goal Mentor

An AI-powered investment education assistant built with LangGraph and Streamlit.

## Table of Contents
- [Team Members](#team-members)
- [Setup](#setup)
- [First Iteration](#first-iteration)
- [Second Iteration](#second-iteration)
- [Peer Review Feedback](#peer-review-feedback)

## Team Members
- Abhishek Harish Thumar
- Kyle Lobo
- Shreyash Naidu Mamidi

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
- Added "Ask Mentor" glossary-based learning support (stores explained terms)


**Changes Made:**
1. Updated GoalState to store task completion status
2. Added progress metrics and completion bar in UI
3. Added task button logic to mark as complete and auto-advance
4. Added glossary sidebar to store terms + explanations
5. Integrated "Ask Mentor" section to explain finance terms in-app

**Test Results:**
```bash
python -m unittest tests/test_user_story_3.py -v
```
- ✅ Scenario 1: Progress bar increases correctly as tasks are completed (33% at 5/15 tasks)
- ✅ Scenario 2: Completing a task activates the next task in sequence
- ✅ Scenario 3: Phase 2 tasks remain locked until all Phase 1 tasks are completed

**UI Features:**
- Strike-through with ✅ indicator for completed tasks
- Automatic "next task" activation to guide learning flow
- Sequential phase unlocking to maintain structured learning
- Glossary panel to review previously explained financial terms
- "Explain This" assistant input for real-time concept clarification


## Running Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Second Iteration

### User Story 4: Risk Tolerance Configuration 
**Issue:** [#4](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/4)

**Implementation:**
- Risk-based investment filtering (Conservative, Moderate, Aggressive)
- Visual risk tolerance indicators with color coding (🟢🟡🔴)
- "Get Risk-Based Recommendations" button generates personalized investment options
- Learning path integration filters tasks by risk appropriateness
- Dynamic recommendations update when risk tolerance changes
- All 3 scenarios implemented and tested

**Changes Made:**
1. `get_investment_recommendations()` - New function to generate risk-appropriate investment guidance
2. `decompose_goal()` - Updated prompt to filter learning tasks based on risk tolerance
3. `init_session_state()` - Added `investment_recommendations` to session state
4. Sidebar - Added visual risk indicators and "Get Risk-Based Recommendations" button
5. Main content - Added investment recommendations display section with clear functionality

**Test Results:**
```bash
pytest tests/test_user_story_4.py -v
```
- ✅ Scenario 1: Conservative risk tolerance - recommends bonds, dividend stocks, index funds
- ✅ Scenario 2: Aggressive risk tolerance - includes growth stocks, sector ETFs, high-volatility assets
- ✅ Scenario 3: Risk tolerance change (Conservative→Moderate) - updates to balanced growth strategies

**UI Features:**
- Color-coded risk tolerance indicator (Conservative 🟢, Moderate 🟡, Aggressive 🔴)
- Risk-appropriate investment recommendations with explanations
- Dynamic filtering of learning tasks based on risk level
- Recommendations persist in session until cleared
- Clear distinction between conservative/moderate/aggressive investment approaches

## Running Tests
```bash
# User Story 4: Risk Tolerance
pytest tests/test_user_story_4.py -v
```

### User Story 5: Risk-Based Portfolio Adjustment (Human-in-the-Loop)
**Issue:** [#5](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/5)

**Implementation:**
- Portfolio allocation suggestions based on risk tolerance (Conservative/Moderate/Aggressive)
- User feedback processing to adjust allocations ("I want more stocks")
- Side-by-side portfolio comparison with different risk levels
- Automatic explanation of allocation changes and trade-offs
- Allocation history tracking for comparison
- All 3 scenarios implemented and tested

**Changes Made:**
1. `suggest_portfolio_allocation()` - Generates risk-appropriate portfolio allocations with AI
2. `explain_allocation_change()` - Explains differences between old and new allocations
3. `adjust_allocation_by_feedback()` - Processes natural language user feedback to modify portfolios
4. `compare_portfolios()` - Compares two portfolios with risk/return analysis
5. `init_session_state()` - Added `suggested_allocation`, `portfolio_comparison`, `allocation_history` to session state
6. Sidebar - Added "Suggest Allocation" button and user feedback adjustment interface
7. Main content - Added allocation display with metrics, comparison selector, and side-by-side view

**Test Results:**
```bash
pytest tests/test_user_story_5.py -v
```
- ✅ Scenario 1: Risk change (Conservative→Moderate) updates allocation (70% bonds→50% stocks/40% bonds)
- ✅ Scenario 2: User feedback ("I want more stocks") adjusts portfolio with trade-off explanation
- ✅ Scenario 3: Alternative portfolio comparison (Moderate vs Aggressive) shows differences and risk metrics

**UI Features:**
- "Suggest Allocation" button generates risk-based portfolio recommendations (Conservative: 70% bonds, Moderate: 50% stocks, Aggressive: 70% stocks)
- User feedback input to adjust allocations in natural language ("I want more stocks", "Less bonds")
- Compare current allocation with alternative risk levels (dropdown + compare button)
- Side-by-side portfolio comparison with detailed risk/return analysis
- Automatic trade-off explanations for all adjustments (volatility vs returns)
- Allocation history tracking for reviewing past suggestions
- Clear buttons for allocation and comparison displays
- Real-time allocation totals with validation (must sum to 100%)

## Running Tests
```bash
pytest tests/test_user_story_5.py -v
```

### User Story 6: Explain "Why?" Behind Recommendations
**Issue:** [#6](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/6)

**Implementation:**
- Added an AI explanation engine for answering "why" questions about portfolio or learning recommendations  
- System explains reasoning based on user profile (age, risk tolerance, time horizon, experience)
- Supports explaining:
  - Portfolio allocations (e.g., "Why 30% bonds instead of 20%?")
  - Risk-based exclusions (e.g., "Why avoid cryptocurrency?")
  - Learning path ordering (e.g., "Why learn bonds before stocks?")
- Uses investment principles (diversification, volatility, compounding, risk/return) in explanations
- Designed for short, clear, beginner-friendly reasoning
- All 3 scenarios implemented and tested

**Changes Made:**
1. `explain_recommendation()` — NEW core function to explain recommendations using user info + previous outputs  
2. Added structured reasoning prompt referencing allocation, recommendations, and learning path  
3. Session state support for storing last allocation, last recommendations, and goal state for explanation  
4. Modular design for reuse across different recommendation types (portfolio or learning tasks)  
5. Optional UI additions for "Why?" questions and reasoning display  

**Test Results:**
```bash
python -m unittest tests/test_user_story_6.py -v
```
- ✅ Scenario 1: Allocation explanation (why 30% bonds vs 20%)
- ✅ Scenario 2: Explanation for avoiding cryptocurrency tailored to risk tolerance
- ✅ Scenario 3: Learning path reasoning (why study bonds before stocks)

**UI Features:**
- "Explain This" input box for asking any "why" question connected to prior recommendations
- AI-generated reasoning displayed in a clean formatted explanation card
- Explanations automatically connect to user profile (age, risk tolerance, time horizon, experience)
- Works for portfolio suggestions, investment warnings, and learning path sequences
- Responses limited to ~200 words for readability
- Allocation history tracking for reviewing past suggestions
- Fully integrated with existing sidebar + main content workflow
- Real-time allocation totals with validation (must sum to 100%)

## Running Tests
```bash
pytest tests/test_user_story_6.py -v
```

### User Story 7: Reset Memory & Start Fresh  
**Issue:** [#7](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/7)

**Implementation:**
- Added full session reset system allowing users to clear all stored financial data, reasoning, and recommendations  
- Supports two reset modes:
  - **Reset Memory:** Clears bot-generated outputs while keeping UI inputs
  - **Start New Session:** Completely resets the user's financial profile and conversation state
- Ensures no previous context influences future recommendations or explanations  
- Designed to prevent contamination from earlier incorrect or experimental inputs  
- All 3 scenarios implemented and tested

**Changes Made:**
1. `reset_all_memory()` — clears learning paths, portfolio analysis, recommendations, reasoning, allocations, and glossary  
2. `reset_profile_and_memory()` — full wipe of user profile state (age, risk tolerance, savings, 401k, stocks, experience, time availability)  
3. Updated main UI to include both buttons with confirmation prompts (Reset Memory + Start New Session)  
4. Session state cleanup redesigned to ensure no stale keys remain after a reset  
5. Improved initialization logic to safely repopulate defaults after full reset  

**Test Results:**
```bash
python -m unittest tests/test_user_story_7.py -v
```
- ✅ Scenario 1: Reset Memory clears all stored bot outputs
- ✅ Scenario 2: Start New Session resets all profile fields to defaults
- ✅ Scenario 3: Incorrect data is successfully removed, ensuring clean re-entry

**UI Features:**
- "🧹 Reset Memory" button clears recommendations, reasoning, allocations, glossary, and portfolio analysis
- "🔄 Start New Session" fully resets all user profile fields plus memory
- Clean confirmation messages guiding users through reset actions
- Ensures recommendations never reference stale or incorrect data
- Safe, isolated session state preventing data leakage between sessions
- Integrated seamlessly into sidebar profile section for quick access
- Real-time allocation totals with validation (must sum to 100%)

## Running Tests
```bash
pytest tests/test_user_story_7.py -v
```

### User Story 8: Historical Data and Empirical Trends
**Issue:** [#8](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/8)

**Implementation:**
- Historical performance data for multiple asset classes with multi-timeframe analysis (10, 20, 30 years)
- Market event analyzer for major crises (2008 financial crisis, COVID-19 crash, dot-com bubble)
- Portfolio allocation comparison with historical evidence
- Educational context and market cycle explanations for all data
- Built-in knowledge base with AI fallback for custom queries
- All 3 scenarios implemented and tested

**Changes Made:**
1. `get_historical_performance()` - Retrieves historical returns for asset classes with market cycle context
2. `compare_allocation_performance()` - Compares portfolio allocations using historical data
3. `get_market_event_data()` - Analyzes how different assets performed during major market events
4. `init_session_state()` - Added `historical_data_display` to session state
5. Main content - Added Historical Performance Data section with:
   - Asset class performance selector (Index Funds, S&P 500, Bonds, Stocks)
   - Market event analyzer (2008 Crisis, COVID Crash, Dot-com Bubble)
   - Portfolio comparison tool with historical evidence
   - Display area with context and educational lessons

**Test Results:**
```bash
pytest tests/test_user_story_8.py -v
```
- ✅ Scenario 1: Historical performance query returns 10/20/30 year data with market cycle context
- ✅ Scenario 2: Portfolio allocation comparison (60/40 vs 100% stocks) shows historical evidence
- ✅ Scenario 3: Market event examples (2008, COVID) present asset class behavior with lessons

**UI Features:**
- Dropdown selector for common asset classes (Index Funds, S&P 500, Bonds, Stocks, Real Estate)
- Dropdown selector for major market events (2008 Financial Crisis, COVID-19 Crash, Dot-com Bubble)
- Text input for custom portfolio allocation comparisons (e.g., "60/40" vs "80/20")
- Historical returns displayed with 10, 20, and 30-year averages
- Market event analysis showing stock/bond performance during crises
- Educational context explaining why assets performed as they did
- Key lessons and takeaways from historical data
- Built-in knowledge base for instant common queries
- AI-powered responses for custom asset classes and events
- Past performance disclaimers on all data
- Clear button to remove displayed data

## Running Tests
```bash
# User Story 8: Historical Data
pytest tests/test_user_story_8.py -v
```

### User Story 9: Portfolio Backtesting ✅
**Issue:** [#9](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/9)

**Implementation:**
- Added portfolio backtesting over multiple time periods (5, 10, 20, 30 years)
- Strategy comparison tool with side-by-side performance metrics
- Crisis period analysis for major market events (2008 crisis, COVID-19, dot-com bubble)
- Performance metrics include: total return, average return, volatility, max drawdown, Sharpe ratio
- Year-by-year return breakdown for detailed analysis

**Changes Made:**
1. `backtest_portfolio()` - Simulates historical portfolio performance with volatility
2. `compare_backtest_strategies()` - Compares two strategies with AI-powered analysis
3. `backtest_crisis_period()` - Shows how portfolio performs during specific crises
4. Session state - Added `backtest_results`, `strategy_comparison`, `crisis_backtest`
5. UI - Three-tab interface (Backtest Portfolio, Compare Strategies, Crisis Performance)

**Test Results:**
```bash
python -m tests/test_user_story_9.py
```
- ✅ Scenario 1: Backtest portfolio with 10-year annual returns and drawdowns
- ✅ Scenario 2: Compare 80/20 vs 60/40 strategies with performance metrics
- ✅ Scenario 3: Portfolio performance during 2008 crisis with explanations
- ✅ Additional: COVID-19 crash, dot-com bubble, multiple time periods

### User Story 10: Account Information and Portfolio Snapshot ✅
**Issue:** [#10](https://github.com/vcu-cmsc-damevski/project-fastcar777/issues/10)

**Implementation:**
- Save portfolio snapshots with custom names (persists to disk)
- Load previously saved portfolios after app restart
- Manage multiple saved portfolios (view, load, delete)
- Export portfolios to .portfolio files for backup/sharing
- Import portfolios from .portfolio files
- Snapshots include: profile, portfolio allocation, learning progress, glossary

**Changes Made:**
1. `save_portfolio_snapshot()` - Saves session state to JSON file on disk
2. `load_all_saved_portfolios()` - Loads all saved portfolios from disk
3. `delete_portfolio_snapshot()` - Deletes portfolio file from disk
4. `apply_pending_load()` - Applies portfolio load before widgets are created
5. `serialize_snapshot()` / `deserialize_snapshot()` - Export/import functionality
6. File storage - `saved_portfolios/` directory for persistent storage
7. Session state - `_pending_load` for two-step load process, `load_message` for notifications
8. UI - Save/Load section in sidebar with portfolio management and export/import

**Test Results:**
```bash
python tests/test_user_story_10.py
```
- ✅ Scenario 1: Save portfolio with holdings and learning progress to disk
- ✅ Scenario 2: Load previous session after restart, restoring all details
- ✅ Scenario 3: Manage multiple portfolios (select, load, delete from disk)
- ✅ Additional: Persistence across sessions, serialization, empty state handling

## Peer Review Feedback

### Feedback from "Moon Walkers" Team

**Date:** November 15, 2024

**Reviewers:** Aneesh Reddy, Abhinav, Ritvik

**Overall Comments:**
The finance learning bot is working well. We tested the app for about 30 minutes and explored different features. AI responses are helpful and relevant to finance topics.

**What we liked:**
- Portfolio analysis gives detailed feedback
- The backtesting feature is interesting, we could see historical performance
- AI explains concepts clearly when asked
- Save/load feature worked when we closed and reopened

**Issues we found:**
1. **Too much in sidebar** - The sidebar is very crowded. We had to scroll a lot to find different options. Maybe put some things in expandable sections?

2. **Multiple input boxes confusing** - There are 3 different text boxes to type in. We weren't sure which one to use when. The examples help but still confusing.

3. **Profile section** - Lot of fields to fill in the profile. Some of these can be optional or hidden until needed.

4. **Main page is too long** - Had to scroll down a lot to see all features. Better organization would help.

**Suggestions:**
- [x] Add some help text or tooltips for finance terms
- [x] Make the UI more compact, too much scrolling right now
- [x] The color coding for risk (🟢🟡🔴) is good, use more visuals like that

---

### Feedback from "Smart Study Assistant Bot" Team

**Date:** November 15, 2024

**Reviewers:** Preetham Reddy Tekula, Balcharan Reddy Adla, Ajay Kumar Reddy Jambula, Varsha Police

**Overall:**
Good finance education tool with many features. The AI gives helpful responses about investments and portfolios. We spent around 45 minutes testing different parts.

**What worked well:**
- Learning path for financial goals is useful
- Portfolio recommendations based on risk tolerance
- Historical data section has good information
- Bot explains things in simple language
- Could save our portfolio and load it back

**Problems we noticed:**
1. **Sidebar has too many options** - We kept opening different expandable sections looking for features. Takes time to understand where everything is.

2. **Three input boxes** - Not clear which input to use for what. Should be one main input that's smarter.

3. **Portfolio entry** - Entering percentages for all 6 asset types one by one is bit tedious. Works fine but could be smoother.

4. **First time using** - When we opened first time, wasn't clear where to start or what to do. Some guidance would help.

**Our suggestions:**
- [x] Combine the input sections or make it clear which one to use when
- [x] Organize sidebar better, group related things
- [x] Add a quick start guide or tutorial for first time
- [x] More visual indicators like the risk tolerance colors
- [ ] Maybe highlight the main features somewhere

**Good points:**
- Backtesting with historical data is cool feature
- Save/load worked smoothly
- AI gives good explanations without too much jargon
- Overall very functional app

---

## UI Improvements Implemented (Post Peer Review)

Based on the peer review feedback, we implemented the following improvements:

### Sidebar Organization
- **Consolidated expandable sections**: Grouped related functionality (Additional Profile Details, Investment Tools, Portfolio Analysis, Save/Load Portfolio, Reset Options) into collapsible expanders
- **Always visible core inputs**: Kept essential fields (Age, Risk Tolerance) always visible
- **Reduced visual clutter**: Default collapsed state for advanced features reduces initial cognitive load

### Unified Input Interface
- **Single smart input box**: Combined "Generate Learning Path", "Ask Your Mentor", and "Explain Why" into one unified interface with radio button selection
- **Clear action selection**: Radio buttons (📚 Generate Learning Path | ❓ Explain a Concept | 🤔 Explain Why) make it obvious which mode is active
- **Dynamic placeholders**: Input placeholder text changes based on selected action to guide users
- **Consistent submission**: Single "🚀 Submit" button works for all three modes

### Visual Improvements
- **Better section headers**: Added emoji indicators and clear section titles
- **Improved spacing**: Reduced cramped feeling with better use of whitespace
- **Help text**: Added contextual help text for each input mode
- **Progress indicators**: Enhanced visual feedback for operations

### User Experience
- **Reduced scrolling**: Collapsible sections mean less scrolling to reach important features
- **Clearer feature discovery**: Related features grouped logically (e.g., all investment tools together)
- **Maintained functionality**: All 10 user stories still fully functional with improved accessibility

---

### Summary of Design and Reasoning Changes

The second iteration focused on improving **trust**, **predictability**, and **reasoning transparency** based on peer review feedback and iteration 2 requirements. Key improvements include:

**Interaction Design Enhancements:**
- **Collapsible Sidebar Sections**: Reduced visual clutter by grouping related features in expandable sections
- **Unified Input Interface**: Combined three separate input boxes into one smart interface with radio button selection
- **Enhanced Visual Feedback**: Added progress indicators, loading spinners, and color-coded status messages throughout
- **Contextual Help**: Integrated tooltips, captions, and help text for all major features

**Reasoning and Robustness:**
- **Transparent AI Processing**: Loading spinners with descriptive text show what the AI is doing ("Creating your personalized learning path...", "Analyzing portfolio...")
- **Explainability Features**: User Story 6 "Explain Why" functionality allows users to understand reasoning behind recommendations
- **Error Handling**: Clear error messages with actionable guidance when issues occur
- **Multi-mode Interaction**: Users can choose between passive (Ask questions) and directive (Generate learning path) modes

---

## Trust and Predictability by Design

### Visual Cues for Confidence and Uncertainty

Our system implements multiple visual indicators to help users understand system state and build trust:

#### 1. **Progress Indicators**
- **Learning Path Progress Bar**: Real-time visual representation (0-100%) showing task completion
  ```python
  st.progress(progress / 100)
  st.caption(f"{progress:.0f}% Complete")
  ```
- **Metrics Display**: Side-by-side task counters (Total Tasks | Completed) provide instant status awareness

#### 2. **Loading States with Context**
Users always know what the system is doing through descriptive loading messages:
- `"Creating your personalized learning path..."` - When generating goals
- `"Generating recommendations..."` - When AI is computing investment advice
- `"Analyzing your portfolio..."` - During portfolio health assessment
- `"Backtesting over 10 years..."` - When running simulations
- `"Teaching you finance wisdom..."` - When explaining concepts

**Design Rationale**: Generic spinners create uncertainty. Our contextual messages set clear expectations and reduce anxiety during AI processing.

#### 3. **Color-Coded Status Messages**
- ✅ **Success (Green)**: `st.success("✅ Learning path created successfully!")`
- ⚠️ **Warning (Yellow)**: `st.warning("⚠️ Total: 95% (need 100%)")`
- 💡 **Info (Blue)**: `st.info("💡 Enter your portfolio allocation")`
- ❌ **Error (Red)**: `st.error("❌ Error initializing OpenAI")`

**Design Rationale**: Immediate visual feedback confirms actions and guides correction when needed.

#### 4. **Risk Tolerance Visual Indicators**
```python
risk_colors = {
    "Conservative": "🟢",
    "Moderate": "🟡",
    "Aggressive": "🔴",
}
```
- **Conservative**: 🟢 "Capital preservation, steady income"
- **Moderate**: 🟡 "Balanced growth with moderate risk"
- **Aggressive**: 🔴 "Maximum growth, higher volatility"

Users instantly recognize their risk level through color psychology and emoji cues.

### Making System Behavior Predictable

#### 1. **Structured Labels and Tooltips**
Every interactive element includes clear labels and contextual help:
```python
st.button(
    "📋 Get Risk-Based Recommendations",
    help=f"Get investment options for {risk_tolerance} risk"
)
```

Examples across the app:
- "Total must equal 100% to analyze" - Disables button until condition met
- "📚 Generate Learning Path | ❓ Explain a Concept | 🤔 Explain Why" - Clear mode selection
- "Years: [5, 10, 20, 30]" - Explicit backtesting timeframe options

#### 2. **Progressive Disclosure**
Complex features are hidden in collapsible sections, reducing cognitive load:
- **💼 Additional Profile Details** (Collapsed by default)
- **💡 Investment Tools** (Collapsed by default)
- **📊 Portfolio Analysis** (Collapsed by default)
- **💾 Save/Load Portfolio** (Collapsed by default)
- **🧹 Reset Options** (Collapsed by default)

**Design Rationale**: Users see only what they need initially. Advanced features are discoverable but not overwhelming.

#### 3. **Input Validation with Real-Time Feedback**
Portfolio allocation provides instant validation:
```python
total_allocation = stocks + bonds + cash + real_estate + crypto + other
if total_allocation == 100:
    st.success(f"✅ Total: {total_allocation}%")
elif total_allocation > 0:
    st.warning(f"⚠️ Total: {total_allocation}% (need 100%)")
```

Users receive immediate feedback on input correctness before submission.

#### 4. **Consistent Action Confirmation**
Every major action includes:
1. **Loading indicator** with descriptive text
2. **Success message** confirming completion
3. **Result display** showing output

Example flow:
```
User clicks "Analyze Portfolio"
→ "Analyzing your portfolio..." (spinner)
→ "✅ Analysis complete!" (success)
→ Portfolio analysis displayed
```

### Confidence Communication

#### 1. **Disclaimer on Historical Data**
```
*Note: Past performance doesn't guarantee future results. 
These are approximate historical averages.*
```

System acknowledges uncertainty in predictions, building trust through honesty.

#### 2. **Explanation Cards with Attribution**
When AI provides reasoning, it's clearly marked:
```html
🧠 Explanation (Why this recommendation?)
```

Users always know when content is AI-generated vs. historical data.

#### 3. **Metrics with Tooltips**
Complex metrics include help text:
```python
st.metric(
    "Sharpe Ratio", 
    f"{sharpe_ratio:.2f}",
    help="Risk-adjusted return (higher is better)"
)
```

Users aren't left guessing what numbers mean.

---

## Reasoning and Robustness Enhancements

### Handling Vague or Conflicting Goals

#### Current Implementation
Our unified input interface addresses vague goals through **mode-specific prompting**:

```python
if action_type == "📚 Generate Learning Path":
    placeholder = "Example: Learn about stock market investing, Understand retirement planning..."
    label_text = "Enter your financial learning goal"
elif action_type == "❓ Explain a Concept":
    placeholder = "Example: What is a dividend yield? What are bonds?"
    label_text = "Ask about any financial term or concept"
else:  # "🤔 Explain Why"
    placeholder = 'Examples: "Why 30% bonds?", "Why not cryptocurrency?"'
    label_text = "Ask why the bot recommended something"
```

**How it works:**
1. User selects mode via radio buttons
2. Placeholder text adapts to show relevant examples
3. Input validation ensures text is provided before submission
4. If goal is too vague, AI generates a generic starting point that can be refined

**Example**: User enters "help with money" → System generates a broad learning path covering budgeting, saving, and investing basics, then allows progressive refinement.

#### Future Enhancements
- Add explicit clarification prompts: "Your goal seems broad. Do you want to focus on: [Saving] [Investing] [Debt Management]?"
- Implement multi-turn goal refinement before generating learning path
- Add goal templates for common scenarios

### Explainability View: Reasoning Transparency

#### User Story 6: "Explain Why" Feature
Users can ask **why** any recommendation was made:

```python
def explain_recommendation(user_question: str, **kwargs) -> str:
    """Explain *why* a recommendation was made."""
    risk_tolerance = kwargs.get("risk_tolerance")
    age = kwargs.get("age")
    suggested_allocation = kwargs.get("suggested_allocation")
    
    # Builds context from user profile and previous recommendations
    context = f"User Profile: Age {age}, {risk_tolerance} risk tolerance..."
    
    # AI explains reasoning connecting to principles
    prompt = """
    Explain the reasoning behind the recommendation
    - Connect to age, risk tolerance, diversification, time horizon
    - Be concise and educational for beginners
    """
```

**Transparency Mechanisms:**

1. **Profile-Based Reasoning**: System explains decisions based on user's:
   - Age (time horizon for compounding)
   - Risk tolerance (volatility capacity)
   - Experience level (complexity appropriateness)
   - Time available (learning pace)

2. **Historical Evidence**: User Story 8 provides data to support claims:
   ```
   "Index funds have delivered:
   - 10-year average: ~12% annually
   - 20-year average: ~10% annually
   Context: Includes recovery from 2008 and COVID..."
   ```

3. **Portfolio Analysis Explanation**:
   ```python
   """
   Provide a comprehensive portfolio analysis including:
   1. Portfolio Health Assessment
   2. Diversification Analysis
   3. Risk Assessment
   4. Specific Recommendations
   5. Explanation: WHY each recommendation matters
   """
   ```

#### LangGraph Reasoning Flow Visibility

The system uses a **multi-node reasoning pipeline** that users can track:

```python
workflow = StateGraph(GoalState)
workflow.add_node("decompose", decompose_goal)
workflow.add_node("analyze_portfolio", analyze_portfolio)
workflow.add_node("display", display_tasks)
```

**Visible Processing Steps:**
1. "Creating your personalized learning path..." → `decompose_goal()` running
2. "Analyzing your portfolio..." → `analyze_portfolio()` running
3. Result display → `display_tasks()` complete

Users see each reasoning step through loading indicators, building trust in the process.

### Autonomy Control: Passive vs. Directive Modes

#### Three Interaction Modes

Our unified interface provides **user-controlled autonomy levels**:

```python
action_type = st.radio(
    "What would you like to do?",
    options=["📚 Generate Learning Path", "❓ Explain a Concept", "🤔 Explain Why"]
)
```

| Mode | Autonomy Level | System Behavior | User Control |
|------|----------------|-----------------|--------------|
| **📚 Generate Learning Path** | **Directive** | System creates 4-5 step plan, analyzes portfolio, suggests allocations | User provides goal, system executes autonomously |
| **❓ Explain a Concept** | **Passive** | System only answers specific questions asked | User drives all interactions, system is reactive |
| **🤔 Explain Why** | **Reflective** | System explains its own prior decisions | User audits system reasoning, maintains oversight |

#### Autonomy Boundaries

**Where System Acts Autonomously:**
- Generating learning subtasks from broad goals
- Analyzing portfolio health and identifying risks
- Suggesting asset allocations based on risk tolerance
- Backtesting portfolios over historical periods
- Comparing investment strategies

**Where User Maintains Control:**
- Choosing interaction mode (passive vs. directive)
- Marking tasks as complete (no auto-advancement)
- Adjusting allocations via natural language feedback
- Selecting what to save/load (portfolio snapshots)
- Resetting memory and starting fresh sessions

**Critical Design Decision**: Tasks are **never auto-completed**. Users must explicitly mark them as done:
```python
if st.button("✓ Complete", key=f"complete_{idx}"):
    st.session_state.goal_state["subtasks"][idx]["completed"] = True
```

This preserves user agency over learning progress.

#### Adjustable Recommendation Strength

Users can modify AI suggestions through:

1. **Risk Tolerance Adjustment**: Changing from Conservative → Aggressive immediately affects recommendations
2. **Natural Language Feedback**: "I want more stocks" → System adjusts allocation with trade-off explanations
3. **Portfolio Comparison**: "Show me aggressive version" → Side-by-side comparison without commitment

Example of human-in-the-loop adjustment:
```python
def adjust_allocation_by_feedback(current_allocation: dict, user_feedback: str):
    """Adjust portfolio based on user feedback"""
    # User: "I want more stocks"
    # System: Increases stocks, explains volatility trade-off
    # User maintains final decision authority
```

### Reflection on Trust, Predictability, and Autonomy

#### How Changes Affect Trust

1. **Consistent Visual Feedback**: Users trust the system because they always know what it's doing
2. **Honest Uncertainty**: Disclaimers on historical data acknowledge limitations
3. **Explainable Decisions**: "Explain Why" feature makes black-box AI transparent
4. **Error Transparency**: Clear error messages with guidance build trust through honesty

**Trust Metric**: Users can verify every recommendation against historical data (User Story 8) and backtesting (User Story 9).

#### How Changes Affect Predictability

1. **Mode Selection**: Radio buttons make system behavior explicit before interaction
2. **Progressive Disclosure**: Collapsible sections mean users aren't surprised by hidden features
3. **Input Validation**: Real-time feedback prevents invalid states
4. **Consistent Patterns**: Every action follows: Loading → Success/Error → Display

**Predictability Metric**: Users should never wonder "what will happen if I click this?"

#### Autonomy Boundaries

**Balanced Approach:**
- **High Autonomy**: Learning path generation (user wants comprehensive plan)
- **Medium Autonomy**: Portfolio analysis (user provides data, system provides insights)
- **Low Autonomy**: Concept explanation (user asks, system answers)
- **Zero Autonomy**: Task completion (always requires user confirmation)

**Key Principle**: System can **recommend**, **explain**, and **analyze**, but never **act** on user's finances without explicit confirmation.

---


