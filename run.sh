#!/bin/bash
# Script to launch F&O Breakout Screener locally
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py

