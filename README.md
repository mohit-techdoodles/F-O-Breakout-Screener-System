# 📈 NSE F&O Breakout Screener (VAH/VAL, CPR & Camarilla & 52W Extremes)

A quantitative institutional stock screener built in Python and Streamlit designed to identify multi-confluence breakout and breakdown momentum across **NSE F&O permitted stocks**.

---

## 🎯 Screening Strategy & Confluence Logic

### Setup A: Bullish Breakout (Long Setup)
- **Condition 1 (Value Area Breakout)**: `LTP > PD_VAH` (Price trading above yesterday's 70% Value Area High)
- **Condition 2 (52-Week High Proximity)**: `LTP >= 52W_High * 0.95` (Trading within 5% of 52-week high)
- **Condition 3 (CPR Money Zone)**: `LTP > PD_CPR_Top` (Trading above yesterday's Central Pivot Range)
- **Condition 4 (Camarilla Breakout)**: `LTP > Cam_H4` (Confirmed momentum above Camarilla H4 level)

### Setup B: Bearish Breakdown (Short Setup)
- **Condition 1 (Value Area Breakdown)**: `LTP < PD_VAL` (Price trading below yesterday's 70% Value Area Low)
- **Condition 2 (52-Week Low Proximity)**: `LTP <= 52W_Low * 1.05` (Trading within 5% of 52-week low)
- **Condition 3 (CPR Money Zone)**: `LTP < PD_CPR_Bottom` (Trading below yesterday's Central Pivot Range)
- **Condition 4 (Camarilla Breakdown)**: `LTP < Cam_L4` (Confirmed liquidation below Camarilla L4 level)

---

## 📁 Project Structure

```text
F&O Breakout Screener System/
├── app.py                # Main Streamlit dashboard application
├── screener_engine.py    # Data ingestion pipeline & vectorized filter engine
├── volume_profile.py     # Steidlmayer / CME 70% Value Area & POC calculation
├── chart_utils.py        # Institutional Plotly candlestick chart with levels
├── fo_stocks.py          # Active NSE F&O permitted symbols & universe selector
├── requirements.txt      # Python dependencies
├── run.sh                # 1-click launch script for macOS / Linux
├── .gitignore            # Git exclusion rules
└── README.md             # Complete documentation
```

---

## 🚀 Running Locally (Mac / Linux)

### Option 1: Using the 1-Click Script
Open Terminal and run:
```bash
cd "/Users/mohitpatel/Desktop/python/F&O Breakout Screener System"
./run.sh
```

### Option 2: Manual Terminal Execution
```bash
cd "/Users/mohitpatel/Desktop/python/F&O Breakout Screener System"
source venv/bin/activate
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ How to Deploy to Streamlit Cloud (100% Free)

Streamlit Community Cloud lets you host this app on a public URL for free so you can access it on any computer or mobile phone.

### Step 1: Initialize Git and Commit
In your terminal:
```bash
cd "/Users/mohitpatel/Desktop/python/F&O Breakout Screener System"
git init
git add .
git commit -m "Initial commit: F&O Breakout Screener"
```

### Step 2: Push to GitHub
1. Go to [GitHub.com](https://github.com) and click **New Repository**.
2. Name it (e.g., `fno-breakout-screener`). Leave it Public (or Private).
3. Copy the push commands from GitHub:
```bash
git remote add origin https://github.com/<your-username>/fno-breakout-screener.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **Create app** > **Deploy a public app from GitHub**.
3. Select your repository: `fno-breakout-screener`.
4. Branch: `main`
5. Main file path: `app.py`
6. Click **Deploy!**

In ~1 minute, your screener will be live on a URL like `https://fno-breakout-screener.streamlit.app`!

