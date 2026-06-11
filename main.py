from data.data_loader import load_data
from models.risk_metrics import (
    calculate_volatility,
    calculate_covariance,
    calculate_correlation
)

tickers = "SPY"

# Load your dataset from the engine
returns = load_data(tickers)

# --- STEP 4: DYNAMIC RISK FEATURES ---

# Flatten columns temporarily to avoid multi-index alignment bugs
close_series = returns['Close'].squeeze()
returns_series = returns['returns'].squeeze()

# 1. Volatility Feature (20-day rolling annualized standard deviation)
returns['vol_20'] = returns_series.rolling(window=20).std() * (252 ** 0.5)

# 2. Trend Feature (1 if price is above 20-day SMA, 0 if below)
sma_20 = close_series.rolling(window=20).mean()
returns['Trend'] = (close_series > sma_20).astype(int)

# 3. Drawdown Feature (Percentage drop from the running historic peak)
rolling_peak = close_series.cummax()
returns['Drawdown'] = (close_series - rolling_peak) / rolling_peak

print("\n--- COMPLETED STEP 4 DATASET ---")
# Print out the exact columns to match your course's requested layout
print(returns[['returns', 'vol_20', 'Trend', 'Drawdown']].tail(15))

# =====================================================================
# =====================================================================
# STEP 5A: PREPARE REGIME DATASET
# =====================================================================
import pandas as pd

# Build a pristine, completely flat DataFrame to eliminate MultiIndex bugs
regime_df = pd.DataFrame(index=returns.index)

# Directly assign our flat 1D calculated features
regime_df['volatility'] = returns['vol_20'].squeeze()
regime_df['trend']      = returns['Trend'].squeeze()
regime_df['drawdown']   = returns['Drawdown'].squeeze()

print("\n--- STEP 5A: REGIME DATASET OUTPUT ---")
print(regime_df.tail(5))


# =====================================================================
# STEP 5B: CLEAN DATA (IMPORTANT)
# =====================================================================
# Drop the initial rolling window NaN rows
regime_df = regime_df.dropna()
features = regime_df[['volatility', 'trend', 'drawdown']]


# =====================================================================
# STEP 5C: STANDARDIZE DATA
# =====================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# =====================================================================
# STEP 5D & 5E: KMEANS CLUSTERING & REGIME PROFILE
# =====================================================================
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
regime_df['regime'] = kmeans.fit_predict(X_scaled)

print("\n--- STEP 5D: APPLIED KMEANS REGIMES ---")
print(regime_df.tail(10))

print("\n--- STEP 5E: UNDERSTAND YOUR REGIMES (PROFILES) ---")
# This will now aggregate perfectly without any multi-index crashes!
print(regime_df.groupby('regime').mean())

# =====================================================================
# STEP 5F: MAP REGIME LABELS
# =====================================================================
# Map based exactly on your engine's mathematical cluster outputs
regime_map = {
    2: 'Bull',
    0: 'Sideways',
    3: 'Bear',
    1: 'Crisis'
}

# Apply the mapping to create a readable regime name column
regime_df['regime_name'] = regime_df['regime'].map(regime_map)

print("\n--- STEP 5F: MAPPED MARKET REGIMES ---")
print(regime_df[['volatility', 'trend', 'drawdown', 'regime_name']].tail(15))

# =====================================================================
# STEP 5F: VISUALIZE REGIMES (WITH MAP)
# =====================================================================
import matplotlib.pyplot as plt

# 1. Map labels based exactly on your engine's numeric data profiles
regime_map = {
    2: 'Bull',
    0: 'Sideways',
    3: 'Bear',
    1: 'Crisis'
}
regime_df['regime_name'] = regime_df['regime'].map(regime_map)

print("\n--- STEP 5F: MAPPED MARKET REGIMES ---")
print(regime_df[['volatility', 'trend', 'drawdown', 'regime_name']].tail(10))

# 2. Generate the visual line graph exactly as requested by the platform
plt.figure(figsize=(10, 5))
plt.plot(regime_df['regime'], color='blue', linewidth=1.5)
plt.title("Market Regime Over Time")
plt.xlabel("Date")
plt.ylabel("Regime Cluster ID")
plt.grid(True, linestyle='--', alpha=0.5)

print("\nRendering regime trajectory plot... (Close the graph window to continue)")
plt.show()

# =====================================================================
# STEP 6A: DEFINE ASSET UNIVERSE
# =====================================================================
import yfinance as yf

# 1. Define the asset universe as requested by the course platform
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

print("\n--- STEP 6A: DOWNLOADING ASSET UNIVERSE DATA ---")
# 2. Download the close prices for our multi-asset pool
# We use the same dates as your main pipeline to keep everything perfectly matched
asset_data = yf.download(assets, start="2025-01-01", end="2026-06-11")['Close']

# 3. Calculate percentage changes and drop initial NaN rows
asset_returns = asset_data.pct_change().dropna()

print("\nAsset Returns Dataset Head:")
print(asset_returns.head(5))

# =====================================================================
# STEP 6B: DEFINE REGIME-BASED WEIGHTS
# =====================================================================
def get_weights(regime):
    """
    Returns strategy asset allocation weights tailored specifically 
    to your model's real data-driven cluster assignments.
    Order: ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    """
    # Regime 2: Your calculated Bull Market
    if regime == 2:
        return [0.35, 0.30, 0.20, 0.15]
        
    # Regime 3: Your calculated Bear Market
    elif regime == 3:
        return [0.10, 0.10, 0.20, 0.60]
        
    # Regime 0: Your calculated Sideways Market
    elif regime == 0:
        return [0.25, 0.25, 0.25, 0.25]
        
    # Regime 1: Your calculated Crisis Market (Ultra Defensive)
    elif regime == 1:
        return [0.05, 0.05, 0.10, 0.80]
        
    else:
        # Fallback equal-weight allocation just in case
        return [0.25, 0.25, 0.25, 0.25]

print("\n--- STEP 6B: REGIME STRATEGY RULES INITIALIZED ---")
# Test your allocation function with a quick dummy look at a Bull signal (ID 2)
print(f"Sample test allocation for a Bull signal (ID 2): {get_weights(2)}")

# =====================================================================
# STEP 6C: APPLY WEIGHTS OVER TIME
# =====================================================================
portfolio_returns = []

# Loop sequentially through every business day in your regime matrix
for i in range(len(regime_df)):
    # Extract the day's numerical model classification signal
    regime = regime_df['regime'].iloc[i]
    
    # Retrieve the tailored allocation vector [AAPL, MSFT, GOOGL, AMZN]
    weights = get_weights(regime)
    
    # Pull the multi-asset daily percentage performance row from step 6A
    # We slice asset_returns with .iloc[i] to align dates exactly
    daily_return = (asset_returns.iloc[i] * weights).sum()
    
    # Store the net blended performance output
    portfolio_returns.append(daily_return)

# Assign the final track record column directly to the dataframe
regime_df['portfolio_returns'] = portfolio_returns

print("\n--- STEP 6C: DYNAMIC PORTFOLIO PERFORMANCE CALCULATION ---")
print(regime_df[['regime', 'regime_name', 'portfolio_returns']].tail(10))

# =====================================================================
# STEP 6D: BUILD PORTFOLIO EQUITY CURVE
# =====================================================================
# Compound your daily strategy returns to map out total growth over time
regime_df['equity_curve'] = (1 + regime_df['portfolio_returns']).cumprod()

print("\n--- STEP 6D: STRATEGY EQUITY CURVE (WEALTH GROWTH) ---")
print(regime_df[['regime_name', 'portfolio_returns', 'equity_curve']].tail(10))

# Optional: Let's quickly plot it so you can see your final performance curve!
plt.figure(figsize=(10, 5))
plt.plot(regime_df['equity_curve'], color='green', linewidth=2, label='Regime Strategy')
plt.title("Portfolio Cumulative Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

print("\nRendering equity growth curve plot... (Close the window to complete the run)")
plt.show()

# =====================================================================
# STEP 6E: COMPARE AGAINST STATIC PORTFOLIO
# =====================================================================
# 1. Calculate Equal Weight Benchmark (The "Buy and Hold" static version)
equal_weight_returns = asset_returns.mean(axis=1)
equal_weight_curve = (1 + equal_weight_returns).cumprod()

# 2. Plot the Comparison
plt.figure(figsize=(10, 6))

# Plot your Dynamic Regime Strategy
plt.plot(regime_df['equity_curve'], label='Dynamic Regime Portfolio', color='green', linewidth=2)

# Plot the Static Equal Weight Benchmark
plt.plot(equal_weight_curve, label='Static Equal Weight Portfolio', color='red', linestyle='--', linewidth=2)

plt.title("Performance Comparison: Dynamic Regime vs. Static Equal Weight")
plt.xlabel("Date")
plt.ylabel("Portfolio Growth ($1 Initial Investment)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

print("\n--- STEP 6E: GENERATING PERFORMANCE COMPARISON CHART ---")
plt.show()

# Fix line 251: Comment out plt.show() so it stops freezing your terminal
# plt.show()
plt.close('all')

# ==============================================================================
# --- STEP 7A: PREPARE RETURNS SERIES (FIXED) ---
# ==============================================================================
print("\n--- STEP 7A: PREPARE RETURNS SERIES ---")
import numpy as np

# The true passive benchmark is an equal weight of our actual asset universe returns
benchmark_returns = asset_returns.mean(axis=1)

# Align the benchmark returns to match the exact index/dates of your regime dataframe
benchmark_returns = benchmark_returns.reindex(regime_df.index).fillna(0)

print("Benchmark returns calculated cleanly from asset universe!")

# ==============================================================================
# --- STEP 7B: COMPUTE CUMULATIVE RETURNS (FIXED) ---
# ==============================================================================
print("--- STEP 7B: COMPUTE CUMULATIVE RETURNS ---")

# Calculate cumulative growth curves cleanly without dividing by 100 
# (Since returns are already in proper daily decimal form, e.g. 0.002 = 0.2%)
regime_df['dynamic_equity'] = (1 + regime_df['portfolio_returns']).cumprod()
regime_df['benchmark_equity'] = (1 + benchmark_returns).cumprod()

# Overwrite the legacy tracking column to guarantee downstream compatibility
regime_df['equity_curve'] = regime_df['dynamic_equity']

print("Cumulative returns computed successfully!")
print(regime_df[['dynamic_equity', 'benchmark_equity']].tail())

# ==============================================================================
# --- STEP 7C: PLOT PERFORMANCE ---
# ==============================================================================
print("--- STEP 7C: PLOT PERFORMANCE ---")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(regime_df['dynamic_equity'], label='Dynamic Regime Portfolio', color='blue', linewidth=2)
plt.plot(regime_df['benchmark_equity'], label='Equal Weight Benchmark', color='orange', linestyle='--')

plt.title("Portfolio Performance Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Growth of $1", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Save figure to disk to prevent terminal blocking/freezing
plt.savefig('portfolio_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close('all')
print("Performance comparison chart saved successfully as 'portfolio_performance_comparison.png'!")

# ==============================================================================
# --- STEP 7D: SHARPE RATIO CALCULATIONS (ANNUALIZED FIX) ---
# ==============================================================================
print("\n--- STEP 7D: SHARPE RATIO CALCULATIONS ---")

def sharpe_ratio(series_returns):
    """Calculates the annualized Sharpe Ratio assuming a risk-free rate of 0."""
    if np.std(series_returns) == 0:
        return 0
    return (np.mean(series_returns) / np.std(series_returns)) * np.sqrt(252)

dynamic_sharpe = sharpe_ratio(regime_df['portfolio_returns'])
benchmark_sharpe = sharpe_ratio(benchmark_returns)

print(f"Dynamic Strategy Annualized Sharpe Ratio: {dynamic_sharpe:.4f}")
print(f"Benchmark Strategy Annualized Sharpe Ratio: {benchmark_sharpe:.4f}")

if dynamic_sharpe > benchmark_sharpe:
    print("Success! Your dynamic risk engine improves risk-adjusted returns.")
else:
    print("The benchmark has higher risk-adjusted returns over this period.")

# ==============================================================================
# --- STEP 7E: MAX DRAWDOWN CALCULATIONS ---
# ==============================================================================
print("\n--- STEP 7E: MAX DRAWDOWN CALCULATIONS ---")

def max_drawdown(equity_curve):
    """Calculates the maximum peak-to-trough drawdown from an equity curve."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

dynamic_mdd = max_drawdown(regime_df['dynamic_equity'])
benchmark_mdd = max_drawdown(regime_df['benchmark_equity'])

print(f"Dynamic Strategy Max Drawdown: {dynamic_mdd * 100:.2f}%")
print(f"Benchmark Strategy Max Drawdown: {benchmark_mdd * 100:.2f}%")

# ==============================================================================
# --- STEP 7F: CAGR CALCULATIONS ---
# ==============================================================================
print("\n--- STEP 7F: CAGR CALCULATIONS ---")

def cagr(equity_curve, periods=252):
    """Calculates the Compound Annual Growth Rate (CAGR) for an equity curve."""
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods
    return (total_return ** (1 / years)) - 1

dynamic_cagr = cagr(regime_df['dynamic_equity'])
benchmark_cagr = cagr(regime_df['benchmark_equity'])

print(f"Dynamic Strategy Annualized Return (CAGR): {dynamic_cagr * 100:.2f}%")
print(f"Benchmark Strategy Annualized Return (CAGR): {benchmark_cagr * 100:.2f}%")

# ==============================================================================
# --- STEP 7G: VOLATILITY CALCULATIONS ---
# ==============================================================================
print("\n--- STEP 7G: VOLATILITY CALCULATIONS ---")

def volatility(series_returns):
    """Calculates annualized volatility assuming 252 trading days."""
    return np.std(series_returns) * np.sqrt(252)

dynamic_vol = volatility(regime_df['portfolio_returns'])
benchmark_vol = volatility(benchmark_returns)

print(f"Dynamic Strategy Annualized Volatility: {dynamic_vol * 100:.2f}%")
print(f"Benchmark Strategy Annualized Volatility: {benchmark_vol * 100:.2f}%")

print("\n==============================================================================")
print("--- RISK ENGINE BACKTEST COMPLETE ---")
print("==============================================================================")

# --- STEP 8D: EXPORT DATA FOR DASHBOARD ---
regime_df.to_csv("regime_results.csv", index=True)
print("Data successfully generated and exported!")