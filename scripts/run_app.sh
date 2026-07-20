#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
streamlit run steering_factory/app.py
