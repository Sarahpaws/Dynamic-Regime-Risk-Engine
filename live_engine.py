import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import argparse
import schedule
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
 
# 
# CONFIGURATION & GLOBAL WEIGHT STRUCTURES
# 
CONFIG = {
    "spy_ticker":     "SPY",
    "assets":         ["SPY", "QQQ", "TLT", "GLD"],
    "training_start": "2020-01-01",
    "n_clusters":     4,
    "vol_window":     20,
    "model_path":     "frozen_regime_model.pkl",
    "scaler_path":    "regime_scaler.pkl",
    "label_map_path": "regime_label_map.pkl",
    "signal_log":     "regime_results.csv",
    "run_time":       "16:30",
}
REGIME_WEIGHTS = {
    "Bull":     {"SPY": 0.40, "QQQ": 0.50, "TLT": 0.10, "GLD": 0.00},
    "Sideways": {"SPY": 0.25, "QQQ": 0.15, "TLT": 0.30, "GLD": 0.30},
    "Bear":     {"SPY": 0.00, "QQQ": 0.00, "TLT": 0.50, "GLD": 0.50},
    "High Risk":{"SPY": 0.10, "QQQ": 0.00, "TLT": 0.40, "GLD": 0.50}
}
 
# 
# CORE FEATURE ENGINEERING PIPELINE
# 
def build_features(df):
    """
    Computes annualized rolling volatility, trend direction, and historic
    drawdown using consistent feature labels matching training data.
    """
    # 1. Check if we have a MultiIndex column structure
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.levels[0]:
            # If SPY is a sub-column under Close
            if 'SPY' in df['Close'].columns:
                close_series = df['Close']['SPY']
            else:
                close_series = df['Close'].iloc[:, 0]
        else:
            raise KeyError("Could not locate top-level 'Close' columns in MultiIndex.")
    else:
        # 2. Fallback for flat, single-index DataFrames
        if 'Close' in df.columns:
            close_series = df['Close']
        elif 'close' in df.columns:
            close_series = df['close']
        else:
            raise KeyError(f"Could not locate 'Close' columns. Available columns: {list(df.columns)}")
            
    # 3. Clean up formatting and align directly with the original Date index
    close_series = pd.Series(close_series.values.ravel(), index=df.index)
    
    # 4. Calculate technical indicator metrics
    returns_series = close_series.pct_change()
    features = pd.DataFrame(index=df.index)
    
    features['vol_20'] = returns_series.rolling(window=CONFIG["vol_window"]).std() * (252 ** 0.5)
    
    sma_20 = close_series.rolling(window=CONFIG["vol_window"]).mean()
    features['Trend'] = (close_series > sma_20).astype(int)
    
    rolling_peak = close_series.cummax()
    features['Drawdown'] = (close_series - rolling_peak) / rolling_peak
    
    return features.dropna()
 
# 
# OFFLINE CALIBRATION / TRAINING LOGIC
# 
def train_model():
    """
    Runs primary historical training logic on historical index values 
    and serializes structural parameters back to workspace disks.
    """
    print(" Downloading market data to calibrate framework models...")
    spy = yf.download(CONFIG["spy_ticker"], start=CONFIG["training_start"], auto_adjust=True, progress=False)
    
    if spy.empty:
        print(" Error: yfinance returned an empty DataFrame. Check your internet connection.")
        return
 
    features = build_features(spy)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=CONFIG["n_clusters"], random_state=42, n_init=10)
    regimes = kmeans.fit_predict(X_scaled)
    
    regime_series = pd.Series(regimes, index=features.index)
    regime_map = auto_label_regimes(regime_series, features)
    
    joblib.dump(kmeans, CONFIG["model_path"])
    joblib.dump(scaler, CONFIG["scaler_path"])
    joblib.dump(regime_map, CONFIG["label_map_path"])
    
    print(" Model calibration complete and frozen to disk!")
    print(f"Regime label map generated: {regime_map}")
 
def auto_label_regimes(regime_series, features):
    """
    Maps categorical labels systematically based on cluster volatility footprints.
    """
    mean_vol = features.groupby(regime_series)['vol_20'].mean().sort_values()
    labels = {}
    
    if len(mean_vol) >= 3:
        labels[mean_vol.index[0]] = "Bull"
        labels[mean_vol.index[1]] = "Sideways"
        labels[mean_vol.index[2]] = "Bear"
    if len(mean_vol) == 4:
        labels[mean_vol.index[3]] = "High Risk"
        
    return labels
 
# 
# LIVE REGIME DETECTION & TRACKING GENERATOR
# 
def detect_regime():
    """
    Pulls recent market quotes, generates forward-looking indicators, 
    runs inference, and logs performance calculations to the shared storage row.
    """
    if not (os.path.exists(CONFIG["model_path"]) and os.path.exists(CONFIG["scaler_path"])):
        print(" Error: Missing serialized ML files. Re-run script using --train flag.")
        return
 
    kmeans = joblib.load(CONFIG["model_path"])
    scaler = joblib.load(CONFIG["scaler_path"])
    label_map = joblib.load(CONFIG["label_map_path"]) if os.path.exists(CONFIG["label_map_path"]) else {}
 
    raw_universe = yf.download(CONFIG["assets"], period="60d", auto_adjust=True, progress=False)
    
    # Flatten MultiIndex wrapper matrix if it exists
    if isinstance(raw_universe.columns, pd.MultiIndex):
        close_prices = raw_universe['Close']
    else:
        close_prices = raw_universe
        
    asset_returns = close_prices.pct_change()
    
    spy_data = pd.DataFrame(index=close_prices.index)
    spy_data['Close'] = close_prices[CONFIG["spy_ticker"]]
    
    features_df = build_features(spy_data)
    if features_df.empty:
        print(" Error processing trailing mathematical arrays.")
        return
        
    latest_timestamp = features_df.index[-1]
    latest_features = features_df.iloc[[-1]]
    latest_returns = asset_returns.iloc[-1]
    
    input_vector = latest_features[['vol_20', 'Trend', 'Drawdown']].values
    scaled_vector = scaler.transform(input_vector)
    
    predicted_idx = int(kmeans.predict(scaled_vector)[0])
    active_regime_name = label_map.get(predicted_idx, f"Cluster {predicted_idx}")
    
    weights = REGIME_WEIGHTS.get(active_regime_name, {a: 0.25 for a in CONFIG["assets"]})
    live_portfolio_return = sum(latest_returns[asset] * weights[asset] for asset in CONFIG["assets"])
    
    print(f" Live Inference [{latest_timestamp.strftime('%Y-%m-%d')}]  Regime: {active_regime_name} | Return: {live_portfolio_return*100:.3f}%")
    
    append_live_metrics(latest_timestamp, latest_features, predicted_idx, active_regime_name, live_portfolio_return)
 
def append_live_metrics(timestamp, feature_row, cluster_idx, regime_name, current_return):
    csv_path = CONFIG["signal_log"]
    
    new_row = pd.DataFrame({
        "volatility":        [feature_row['vol_20'].values[0]],
        "trend":             [feature_row['Trend'].values[0]],
        "drawdown":          [feature_row['Drawdown'].values[0]],
        "regime":            [cluster_idx],
        "regime_name":       [regime_name],
        "portfolio_returns": [current_return],
        "equity_curve":      [np.nan],
        "dynamic_equity":    [np.nan],
        "benchmark_equity":  [np.nan]
    }, index=[timestamp])
    new_row.index.name = "Date"
 
    if os.path.exists(csv_path):
        historical_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        historical_df = historical_df[historical_df.index != timestamp]
        
        last_strat_equity = historical_df['dynamic_equity'].iloc[-1] if 'dynamic_equity' in historical_df.columns else 1.0
        last_bench_equity = historical_df['benchmark_equity'].iloc[-1] if 'benchmark_equity' in historical_df.columns else 1.0
        
        new_row['dynamic_equity'] = last_strat_equity * (1.0 + current_return)
        new_row['benchmark_equity'] = last_bench_equity
        
        final_matrix = pd.concat([historical_df, new_row])
    else:
        new_row['dynamic_equity'] = 1.0 * (1.0 + current_return)
        new_row['benchmark_equity'] = 1.0
        final_matrix = new_row
        
    final_matrix.to_csv(csv_path)
    print(" Production storage matrix synchronized.")
 
# 
# RUNTIME INTERFACE HANDLER
# 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Regime Risk Engine Daemon Pipeline")
    parser.add_argument("--train", action="store_true", help="Execute baseline model feature training calibration routines")
    args = parser.parse_args()
    
    if args.train:
        train_model()
    else:
        detect_regime()
        
        schedule.every().day.at(CONFIG["run_time"]).do(detect_regime)
        print(f" Risk engine tracking active. Monitoring market schedules for {CONFIG['run_time']} updates daily...")
        
        while True:
            schedule.run_pending()
            time.sleep(1)