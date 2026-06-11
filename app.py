import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

st.set_page_config(page_title="Institutional Live Risk Monitor Engine", layout="wide")

st.title("📊 Institutional Regime & Risk Monitoring Platform")
st.markdown("**System Status:** 🟢 Cloud Risk Pipeline Active — Live Telemetry Layer")

# Global Allocation Multipliers
REGIME_WEIGHTS = {
    "Bull":     {"SPY": "40%", "QQQ": "50%", "TLT": "10%", "GLD": "0%"},
    "Sideways": {"SPY": "25%", "QQQ": "15%", "TLT": "30%", "GLD": "30%"},
    "Bear":     {"SPY": "0%",  "QQQ": "0%",  "TLT": "50%", "GLD": "50%"},
    "High Risk":{"SPY": "10%", "QQQ": "0%",  "TLT": "40%", "GLD": "50%"}
}

# ─────────────────────────────────────────────────────────────
# CLOUD COMPUTE KERNEL
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300) # Caches data for 5 minutes (300s)
def fetch_and_predict():
    assets = ["SPY", "QQQ", "TLT", "GLD"]
    raw_universe = yf.download(assets, period="120d", auto_adjust=True, progress=False)
    
    close_matrix = raw_universe['Close'] if isinstance(raw_universe.columns, pd.MultiIndex) else raw_universe
    asset_returns = close_matrix.pct_change()
    
    # Feature Engineering
    spy_close = pd.Series(close_matrix['SPY'].to_numpy().ravel(), index=close_matrix.index)
    spy_returns = spy_close.pct_change()
    
    features = pd.DataFrame(index=close_matrix.index)
    features['vol_20'] = spy_returns.rolling(window=20).std() * (252 ** 0.5)
    features['Trend'] = (spy_close > spy_close.rolling(window=20).mean()).astype(int)
    features['Drawdown'] = (spy_close - spy_close.cummax()) / spy_close.cummax()
    features = features.dropna()
    
    # Load Models safely from the root directory
    kmeans = joblib.load("frozen_regime_model.pkl")
    scaler = joblib.load("regime_scaler.pkl")
    label_map = joblib.load("regime_label_map.pkl")
    
    X_scaled = scaler.transform(features[['vol_20', 'Trend', 'Drawdown']].values)
    predictions = kmeans.predict(X_scaled)
    distances = kmeans.transform(X_scaled)
    
    # Build Historical Dataframe
    df_out = pd.DataFrame(index=features.index)
    df_out['regime_name'] = [label_map.get(p, f"Cluster {p}") for p in predictions]
    df_out['volatility'] = features['vol_20']
    df_out['drawdown'] = features['Drawdown']
    df_out['confidence'] = 1 / (1 + distances.min(axis=1))
    
    # Calculate portfolio returns
    portfolio_rets = []
    for idx, row in df_out.iterrows():
        reg = row['regime_name']
        w = {a: float(val.replace('%',''))/100 for a, val in REGIME_WEIGHTS.get(reg).items()}
        ret = sum(asset_returns.loc[idx, a] * w[a] for a in assets) if idx in asset_returns.index else 0
        portfolio_rets.append(ret)
        
    df_out['portfolio_returns'] = portfolio_rets
    df_out['rolling_VaR'] = df_out['portfolio_returns'].rolling(window=60).quantile(0.05)
    
    return df_out

try:
    df = fetch_and_predict()
    
    # Extract Latest Metrics
    current_active_regime = df["regime_name"].iloc[-1]
    current_confidence = df['confidence'].iloc[-1] * 100
    current_var = df['rolling_VaR'].iloc[-1] if not np.isnan(df['rolling_VaR'].iloc[-1]) else -0.015
    latest_update = df.index[-1].strftime('%Y-%m-%d %H:%M')
    
    # Compute Regime Duration
    duration_days = 0
    for past_regime in reversed(df["regime_name"].values):
        if past_regime == current_active_regime:
            duration_days += 1
        else:
            break

    # Operational Traffic Light
    if current_var < -0.030:
        traffic_status, traffic_color, text_color = "🔴 HIGH OPERATIONAL RISK", "#f2dede", "#a94442"
    elif current_var < -0.015:
        traffic_status, traffic_color, text_color = "🟡 MEDIUM OVERWATCH RISK", "#fcf8e3", "#8a6d3b"
    else:
        traffic_status, traffic_color, text_color = "🟢 LOW RISK MATRIX REGIME", "#dff0d8", "#3c763d"

    # ==========================================================================
    # DISPLAY LAYER
    # ==========================================================================
    st.markdown(f"""
        <div style="background-color:{traffic_color}; padding:22px; border-radius:8px; border-left: 10px solid {text_color}; margin-bottom: 25px; color:{text_color};">
            <h3 style="margin:0; font-weight:800;">System Health: {traffic_status}</h3>
            <div style="display: flex; gap: 40px; margin-top: 12px; font-size: 15px;">
                <p style="margin:0;">Active Regime: <strong>{current_active_regime}</strong></p>
                <p style="margin:0;">Centroid Confidence: <strong>{current_confidence:.1f}%</strong></p>
                <p style="margin:0;">Persistence Horizon: <strong>{duration_days} Days</strong></p>
                <p style="margin:0;">Last Update: <code>{latest_update}</code></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric(label="Current 60D Value at Risk (95% VaR)", value=f"{current_var*100:.2f}%")
    col_kpi2.metric(label="Trailing Period Volatility Index", value=f"{(df['volatility'].iloc[-1]*100):.2f}%")
    col_kpi3.metric(label="System Maximum Peak Drawdown", value=f"{(df['drawdown'].min()*100):.2f}%")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 Trailing 60-Day Forward-Risk Evolution Metrics Layer")
        risk_chart = df[['rolling_VaR']].copy().dropna()
        risk_chart.columns = ['95% Value at Risk (VaR)']
        st.area_chart(risk_chart, use_container_width=True)

    with col_right:
        st.subheader("🎯 Actionable Target Allocation Matrix")
        rec_allocations = REGIME_WEIGHTS.get(current_active_regime)
        for target_asset, structural_weight in rec_allocations.items():
            st.markdown(f"🔹 **{target_asset}** Allocation Weight: `{structural_weight}`")

except Exception as e:
    st.error(f"Initialization error: {e}")