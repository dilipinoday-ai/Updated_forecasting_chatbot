# app.py
import os
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta, datetime, date

import holidays
import google.generativeai as genai

# =========================
# 0) STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="🛍️ AI Business Decision Forecaster", layout="wide")
st.title("🛍️ AI-Powered Business Decision Forecaster (Seasonal & Festival Aware)")

# =========================
# 1) GEMINI CONFIG (secrets/.env)
# =========================
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    pass

if not GEMINI_API_KEY:
    # fallback to .env
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_ready = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_ai = genai.GenerativeModel("models/gemini-1.5-flash-002")
        gemini_ready = True
    except Exception as e:
        st.error(f"Gemini init error: {e}")
else:
    st.warning("⚠️ Gemini API key not found. Add it to Streamlit secrets or a .env file.")

# =========================
# 2) FESTIVAL ↔ PRODUCT MAPPING
#    (You can tweak these multipliers anytime)
# =========================
FESTIVAL_PRODUCT_MULTIPLIER = {
    # category -> festival_tag -> demand multiplier ( >1 boosts, <1 suppresses )
    "traditional_clothing": {
        "diwali": 1.35,
        "navratri": 1.40,
        "durga_puja": 1.35,
        "eid": 1.25,
        "raksha_bandhan": 1.30,
        "wedding_season": 1.30,
        "onam": 1.20,
        "pongal": 1.15,
    },
    "winter_clothing": {
        "winter": 1.40,
        "diwali": 1.05,
        "raksha_bandhan": 0.80,  # lower demand around Rakhi for winter wear
        "summer": 0.75,
    },
    "electronics_gadgets": {
        "diwali": 1.30,
        "independence_day": 1.15,
        "new_year": 1.20,
        "back_to_school": 1.10,
        "eid": 1.10,
    },
    "sweets_gifts": {
        "diwali": 1.45,
        "raksha_bandhan": 1.40,
        "eid": 1.35,
        "christmas": 1.25,
        "valentines": 1.25,
        "new_year": 1.25,
    },
    "summer_apparel": {
        "summer": 1.35,
        "monsoon": 0.95,
        "winter": 0.70,
    },
    "school_supplies": {
        "back_to_school": 1.35,
        "summer": 0.85,
        "winter": 0.90,
    }
}

# Helper: quick default suggestions for user
CATEGORY_SUGGESTIONS = [
    "traditional_clothing",
    "winter_clothing",
    "electronics_gadgets",
    "sweets_gifts",
    "summer_apparel",
    "school_supplies",
]

# =========================
# 3) FESTIVAL TAGGING LOGIC
# =========================
def infer_season(d: date) -> str:
    """Simple India-focused season heuristic by month."""
    m = d.month
    if m in [12,1,2]:
        return "winter"
    if m in [3,4,5]:
        return "summer"
    if m in [6,7,8,9]:
        return "monsoon"
    return "autumn"  # Oct-Nov (festive heavy), Dec already handled as winter

def festival_tags_in_range(start_dt: date, end_dt: date):
    """Return a set of 'festival tags' present in the given range."""
    tags = set()
    in_holidays = holidays.India()
    rng = pd.date_range(start=start_dt, end=end_dt, freq="D")
    for d in rng:
        dd = d.date()
        # season
        tags.add(infer_season(dd))
        # known holiday names
        if dd in in_holidays:
            name = in_holidays.get(dd).lower()
            if "diwali" in name or "deepavali" in name:
                tags.add("diwali")
            if "eid" in name:
                tags.add("eid")
            if "holi" in name:
                tags.add("holi")
            if "christmas" in name:
                tags.add("christmas")
            if "independence" in name:
                tags.add("independence_day")
            if "pongal" in name or "makar sankranti" in name:
                tags.add("pongal")
            if "onam" in name:
                tags.add("onam")
            if "navratri" in name:
                tags.add("navratri")
            if "durga puja" in name or "dussehra" in name:
                tags.add("durga_puja")
            if "new year" in name:
                tags.add("new_year")

    # Add heuristic tags by month if not present:
    # Raksha Bandhan usually falls Aug (varies). If range covers Aug, add tag.
    if any(d.month == 8 for d in rng):
        tags.add("raksha_bandhan")

    # Wedding season (Oct–Feb, roughly; plus Apr-May spikes)
    if any(d.month in [10,11,12,1,2,4,5] for d in rng):
        tags.add("wedding_season")

    # Back to school (Jun–Jul)
    if any(d.month in [6,7] for d in rng):
        tags.add("back_to_school")

    return sorted(tags)

def demand_multiplier_for_category(category: str, active_tags: list[str]) -> float:
    base = 1.0
    mapping = FESTIVAL_PRODUCT_MULTIPLIER.get(category, {})
    # multiply effects (cap to avoid extremes)
    for t in active_tags:
        if t in mapping:
            base *= mapping[t]
    # cap multiplier to a sane range
    return float(np.clip(base, 0.5, 2.0))

# =========================
# 4) CORE MODEL FUNCTIONS
# =========================
def preprocess_data(df, date_col, value_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col])
    df = df.sort_values(date_col)
    df = df.reset_index(drop=True)
    return df

def add_holiday_features(df, date_col):
    df = df.copy()
    in_holidays = holidays.India()
    df["is_holiday"] = df[date_col].dt.date.apply(lambda x: x in in_holidays)
    df["season"] = df[date_col].dt.date.apply(infer_season)
    return df

def train_lstm(series: pd.Series, lookback=30):
    values = series.values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1,1))

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(48, return_sequences=True, input_shape=(lookback,1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X.reshape((X.shape[0], X.shape[1], 1)), y, epochs=12, batch_size=16, verbose=0)
    return model, scaler

def lstm_forecast(model, scaler, series: pd.Series, periods: int, lookback=30):
    arr = series.values.astype(float)
    scaled = scaler.transform(arr.reshape(-1,1))
    last_seq = scaled[-lookback:]
    batch = last_seq.reshape(1, lookback, 1)
    out = []
    for _ in range(periods):
        pred = model.predict(batch, verbose=0)[0,0]
        out.append(pred)
        batch = np.append(batch[:,1:,:], [[[pred]]], axis=1)
    inv = scaler.inverse_transform(np.array(out).reshape(-1,1)).flatten()
    return inv

def arima_forecast(series: pd.Series, periods: int, order=(5,1,0)):
    model = ARIMA(series.values.astype(float), order=order)
    fit = model.fit()
    fc = fit.forecast(steps=periods)
    return fc.values

# =========================
# 5) BUY SCORE + REASONS
# =========================
def compute_buy_score(hist_series: pd.Series, forecast_vals: np.ndarray, season_multiplier: float):
    """Return a score (0-100) + components."""
    # growth vs recent mean
    recent_mean = float(np.mean(hist_series.tail(14)))
    future_mean = float(np.mean(forecast_vals[:max(1, min(14, len(forecast_vals)))]))
    growth_pct = 0.0 if recent_mean == 0 else ((future_mean - recent_mean) / recent_mean) * 100.0

    # simple trend using linear fit on last 30 + next 14 (synthetic)
    y_hist = hist_series.tail(30).values.astype(float)
    if len(y_hist) >= 3:
        x_hist = np.arange(len(y_hist))
        slope_hist = np.polyfit(x_hist, y_hist, 1)[0]
    else:
        slope_hist = 0.0

    # combine: growth (weighted 0.6), slope sign (0.2), festival multiplier (0.2)
    # map to 0..100
    growth_score = np.clip(50 + growth_pct, 0, 100)  # growth 0 -> 50 baseline
    slope_score  = 70 if slope_hist > 0 else (30 if slope_hist < 0 else 50)
    mult_score   = np.clip(50 + (season_multiplier - 1.0) * 100, 0, 100)

    total = 0.6*growth_score + 0.2*slope_score + 0.2*mult_score
    return float(total), {
        "recent_mean": recent_mean,
        "future_mean": future_mean,
        "growth_pct": growth_pct,
        "slope_hist": slope_hist,
        "season_multiplier": season_multiplier,
        "growth_score": growth_score,
        "slope_score": slope_score,
        "mult_score": mult_score
    }

def buy_label_from_score(score: float, threshold_buy: float = 58.0, threshold_hold: float = 48.0):
    if score >= threshold_buy:
        return "BUY"
    if score <= threshold_hold:
        return "DON'T BUY"
    return "HOLD / CAUTIOUS"

# =========================
# 6) GEMINI INSIGHTS (with retry for 503)
# =========================
def call_gemini_with_retry(prompt: str, retries=4):
    if not gemini_ready:
        return "Gemini not configured."
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = gemini_ai.generate_content(prompt)
            return resp.text
        except Exception as e:
            msg = str(e).lower()
            if "503" in msg or "unavailable" in msg:
                time.sleep(delay)
                delay *= 2
                continue
            return f"Gemini error: {e}"
    return "Gemini temporarily unavailable after retries."

def gemini_prompt(value_col, product_category, active_tags, buy_label, score, parts, head_rows, forecast_head):
    return f"""
You are a senior retail business analyst AI for India.

DATA:
- Metric: {value_col}
- Product category: {product_category}
- Active seasonal/festival tags in horizon: {active_tags}
- Decision (rule-based): {buy_label} (score: {score:.1f}/100)
- Components: {parts}
- Recent last 10 values: {head_rows}
- Next forecasted 10: {forecast_head}

TASK:
1) Validate/critique the decision (BUY / HOLD / DON'T BUY) with reasoning grounded in data, tags and Indian retail patterns.
2) Explain the key drivers (trend, growth vs recent mean, festival/season multipliers like Diwali, Raksha Bandhan, winter, etc.).
3) Give exactly 3 actionable recommendations (pricing, inventory timing, marketing, assortment).
4) If DON'T BUY or HOLD, suggest alternatives (e.g., for Raksha Bandhan prefer traditional wear, sweets/gifts).
Keep it concise, bullet-style, practical.
"""

# =========================
# 7) SIDEBAR UI
# =========================
with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    model_choice = st.radio("Forecast Model", ["Hybrid (Average)", "LSTM", "ARIMA"])
    st.caption("Hybrid = average of LSTM & ARIMA")

    st.divider()
    st.subheader("🧾 Columns")
    chosen_date_col = st.text_input("Date column name (leave blank to select later)")
    chosen_value_col = st.text_input("Target (sales) column name (leave blank to select later)")

    st.divider()
    st.subheader("🛒 Product Category")
    product_category = st.selectbox(
        "Pick or type:",
        options=CATEGORY_SUGGESTIONS,
        index=0,
        help="Used to apply festival/season demand multipliers."
    )

# =========================
# 8) MAIN
# =========================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ask columns if not given
    date_col = chosen_date_col if chosen_date_col in df.columns else st.selectbox("Select Date Column", options=df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    value_col = chosen_value_col if chosen_value_col in df.columns else st.selectbox("Select Target Column", options=numeric_cols)

    # Forecast horizon
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    min_date, max_date = df[date_col].min().date(), df[date_col].max().date()

    forecast_to = st.date_input(
        "Forecast up to",
        value=(max_date + timedelta(days=30)),
        min_value=max_date + timedelta(days=1),
        max_value=max_date + timedelta(days=365)
    )
    forecast_days = (forecast_to - max_date).days

    if forecast_days <= 0:
        st.warning("Please choose a future date.")
        st.stop()

    if st.button("🚀 Run Forecast & Business Analysis"):
        with st.status("Crunching numbers...", expanded=True) as status:
            try:
                st.write("🧹 Preprocessing & feature enrichment...")
                df_proc = preprocess_data(df, date_col, value_col)
                df_proc = add_holiday_features(df_proc, date_col)

                status.update(label="Training forecast model(s)...")
                # LSTM
                if model_choice in ["LSTM", "Hybrid (Average)"]:
                    if len(df_proc) < 40:
                        st.warning("Not enough rows for LSTM (needs ~40+). Falling back to ARIMA.")
                        lstm_vals = None
                    else:
                        lstm_model, scaler = train_lstm(df_proc[value_col])
                        lstm_vals = lstm_forecast(lstm_model, scaler, df_proc[value_col], forecast_days)
                else:
                    lstm_vals = None

                # ARIMA
                if model_choice in ["ARIMA", "Hybrid (Average)"] or lstm_vals is None:
                    arima_vals = arima_forecast(df_proc[value_col], forecast_days)
                else:
                    arima_vals = None

                # Combine
                if model_choice == "Hybrid (Average)":
                    if lstm_vals is None:  # fallback
                        forecast_values = arima_vals
                    elif arima_vals is None:
                        forecast_values = lstm_vals
                    else:
                        forecast_values = (lstm_vals + arima_vals) / 2.0
                elif model_choice == "LSTM":
                    forecast_values = lstm_vals
                else:
                    forecast_values = arima_vals

                forecast_dates = pd.date_range(start=df_proc[date_col].iloc[-1] + timedelta(days=1), periods=forecast_days)
                fc_df = pd.DataFrame({date_col: forecast_dates, value_col: forecast_values, "Type": "Forecast"})
                hist_df = df_proc[[date_col, value_col]].assign(Type="Historical")
                full_df = pd.concat([hist_df, fc_df], ignore_index=True)

                # Visual
                status.update(label="Rendering charts...")
                fig = px.line(full_df, x=date_col, y=value_col, color="Type", title=f"{value_col} Forecast")
                st.plotly_chart(fig, use_container_width=True)

                # Festival tags present in the forecast window
                active_tags = festival_tags_in_range(forecast_dates.min().date(), forecast_dates.max().date())
                st.write("🎉 Active seasonal/festival tags in horizon:", ", ".join(active_tags))

                # Demand multiplier for the chosen category
                mult = demand_multiplier_for_category(product_category, active_tags)

                # Decision
                score, parts = compute_buy_score(df_proc[value_col], np.array(forecast_values), mult)
                decision = buy_label_from_score(score)

                # Summary box
                st.success(f"Decision: **{decision}**  |  Buy Score: **{score:.1f}/100**")
                with st.expander("See decision components"):
                    st.json(parts)

                # Gemini insight
                if gemini_ready:
                    status.update(label="Getting AI (Gemini) recommendations...")
                    prompt = gemini_prompt(
                        value_col=value_col,
                        product_category=product_category,
                        active_tags=active_tags,
                        buy_label=decision,
                        score=score,
                        parts=parts,
                        head_rows=df_proc[value_col].tail(10).tolist(),
                        forecast_head=list(np.array(forecast_values)[:10])
                    )
                    insight = call_gemini_with_retry(prompt)
                    st.subheader("🤖 AI Business Recommendations")
                    st.write(insight)
                else:
                    st.info("Gemini not configured—skipping AI narrative.")

                status.update(label="Forecast complete!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                status.update(label="Failed", state="error")

else:
    st.info("👋 Upload a CSV to begin (date + numeric sales column required).")

st.markdown("---")
st.caption("© 2025 Business AI Forecaster | LSTM • ARIMA • Seasonality • Gemini AI")
