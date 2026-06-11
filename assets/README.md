# Real-Time Market Regime & Risk Monitoring Platform

## Live Demo

https://dynamic-regime-risk-engine.onrender.com/

## Dashboard Preview

![Dashboard](assets/dashboard.png)

## Overview

A cloud-deployed portfolio risk monitoring system that continuously ingests live market data, identifies market regimes using machine learning, and provides real-time portfolio risk analytics.

## Overview

A cloud-deployed portfolio risk monitoring platform that continuously ingests live market data, detects market regimes using machine learning, and provides real-time risk analytics and portfolio allocation insights.

The platform uses volatility, trend, and drawdown features to classify market conditions and support risk-aware investment decisions.

## Key Features

- Live market data ingestion using Yahoo Finance
- Market regime classification using KMeans clustering
- Volatility, trend, and drawdown feature engineering
- Dynamic portfolio allocation recommendations
- Value-at-Risk (VaR) analytics
- Conditional Value-at-Risk (CVaR) analytics
- Interactive Streamlit dashboard
- Cloud deployment on Render

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Yahoo Finance (yfinance)
- Render

## Architecture

Yahoo Finance Data
↓
Feature Engineering
↓
KMeans Regime Detection
↓
Risk Analytics (VaR/CVaR)
↓
Allocation Engine
↓
Streamlit Dashboard
↓
Render Cloud Deployment

## Market Regimes

The model classifies market conditions into:

- Bull Market
- Bear Market
- Sideways Market
- Crisis / High-Risk Market

These regimes are used to support portfolio allocation and risk monitoring decisions.

## Risk Metrics

The platform monitors:

- Portfolio Volatility
- Value-at-Risk (VaR)
- Conditional Value-at-Risk (CVaR)
- Drawdown Analysis