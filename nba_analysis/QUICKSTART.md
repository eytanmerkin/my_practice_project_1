# 🚀 Quick Start Guide

Get up and running in 3 easy steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all necessary packages including:
- `nba_api` - For fetching NBA data
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning
- `streamlit`, `plotly` - Interactive web app

## Step 2: Run the Analysis

Choose one of these three options:

### Option A: Basic Command-Line Script ⭐ (Recommended to start)

```bash
python pd_wp_analysis.py
```

**What it does:**
- ✅ Fetches 30 seasons of NBA data (~15-20 minutes, ~900 team-seasons for robust analysis)
- ✅ Creates scatter plot showing point differential vs winning %
- ✅ Trains linear regression model
- ✅ Saves visualization as `nba_point_diff_analysis.png`
- ✅ Saves data as `nba_team_seasons_processed.csv`
- ✅ Launches interactive predictor

**Example output:**
```
=============================================================
LINEAR REGRESSION RESULTS
=============================================================
R² Score: 0.9421
RMSE: 0.0453
Equation: Win% = 0.5002 + 0.0332 × Point Diff

Interpretation: Each point of differential = 2.72 wins over 82 games
=============================================================

INTERACTIVE WIN PREDICTOR
Point differential per game: 7

  📈 Predicted Win%: 73.2%
  🏆 Expected Wins: 60.0
  📉 Expected Losses: 22.0
  📊 Record: 60-22
```

### Option B: Streamlit Web App 🌐 (Most Interactive)

```bash
streamlit run streamlit_app.py
```

**Features:**
- 📊 Interactive sliders and inputs
- 📈 Live updating predictions
- 🎯 Hoverable scatter plots
- 📋 Data filtering and exploration
- 💾 Download processed data
- 🧮 Quartile analysis tab

Your browser will open automatically to `http://localhost:8501`

### Option C: Advanced Quartile Predictor 🎯 (Early Season Focus)

First, run Option A to generate the data file, then:

```bash
python quartile_predictor.py
```

**Commands available:**
- `predict` - Predict final wins based on current point differential
- `compare` - Compare actual record vs point differential expectations
- `simulate` - Monte Carlo simulation of season outcomes
- `demo` - Run all example scenarios

**Example interaction:**
```
Command [predict/compare/simulate/demo/quit]: predict
  Games played so far: 20
  Current point differential per game: 5.5

  📊 PREDICTION RESULTS
  Season Progress: 24.4%
  Predicted Final Wins: 51.8
  Expected Record: 52-30
  Win Range: 44-60
  Confidence: 62%
```

## Step 3: Explore!

### Try These Scenarios:

**Elite Team:**
- Point Diff: +10 → ~65 wins

**Playoff Contender:**
- Point Diff: +3 → ~48 wins

**Below Average:**
- Point Diff: -5 → ~23 wins

### Understanding the Results:

| Point Diff | Expected Wins | Win % | Interpretation |
|-----------|---------------|-------|----------------|
| +10       | ~64           | 78%   | Championship contender |
| +7        | ~60           | 73%   | Top seed |
| +5        | ~55           | 67%   | Strong playoff team |
| +3        | ~49           | 60%   | Playoff team |
| 0         | ~41           | 50%   | Average |
| -3        | ~33           | 40%   | Below average |
| -5        | ~28           | 34%   | Lottery team |
| -8        | ~21           | 26%   | Bottom feeder |

## Common Use Cases

### 1. **Evaluate Current Team** (Mid-Season)

After 30 games, your team is 18-12 with +4.5 point differential:

```python
# In streamlit app or quartile predictor:
Games Played: 30
Point Diff: +4.5
→ Projected: 53-29 (solid playoff team)
```

### 2. **Identify Over/Underperformers**

Team is 22-8 (73% win rate) but only +3.5 point differential:

```python
# This suggests they're "lucky" and may regress
Expected win% based on point diff: 61%
Actual win%: 73%
→ Due for regression (expect closer to 50 wins than 60)
```

### 3. **Early Season Projection**

After just 15 games with +7 point differential:

```python
Games: 15 (18% of season)
Point Diff: +7.0
→ Projected: 59-23
Confidence: 59% (still early)
Margin: ±10 wins
```

## Troubleshooting

**Problem:** `FileNotFoundError` when running quartile_predictor.py
```bash
# Solution: Run the base analysis first
python pd_wp_analysis.py
```

**Problem:** API rate limiting
```bash
# Solution: Already handled with 0.6s delays
# If persistent, increase sleep time in pd_wp_analysis.py
```

**Problem:** Installation issues
```bash
# Solution: Use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## What Gets Created

After running the scripts, you'll have:

```
nba_pd_wp/
├── nba_point_diff_analysis.png      # Scatter plot + regression
├── nba_team_seasons_processed.csv   # Full dataset
├── prediction_evolution.png         # (if running quartile demo)
└── [your source files]
```

## Next Steps

1. ✅ **Run the basic analysis** (`python pd_wp_analysis.py`)
2. 🌐 **Try the Streamlit app** (`streamlit run streamlit_app.py`)
3. 📊 **Experiment with predictions** (try different point differentials)
4. 🎯 **Run quartile analysis** (`python quartile_predictor.py`)
5. 📈 **Use for real teams** (input actual data from current season)

## Tips for Best Results

- **More games = better predictions**: Confidence increases throughout season
- **Point differential > record**: Early in season, trust point diff more than W-L
- **Watch for extremes**: Teams with huge gaps between record and point diff will regress
- **Context matters**: Injuries, trades, and schedule can affect projections

---

**Ready to predict some wins? Let's go! 🏀**

Questions? Check the main `README.md` for detailed documentation.

