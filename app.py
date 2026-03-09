# app.py
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

st.set_page_config(page_title="AI Flood Prediction Dashboard", layout="wide")
st.title("🚀 AI Flood Prediction Dashboard with Control & Suggestions")

# -------------------------------
# Basic styling
# -------------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #020617 40%, #0b1120 100%);
        color: #e5e7eb;
    }
    .stApp {
        background-color: transparent;
    }
    .section-card {
        background-color: rgba(15,23,42,0.85);
        padding: 1.25rem 1.5rem;
        border-radius: 0.9rem;
        border: 1px solid #1e293b;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
        margin-bottom: 1.5rem;
    }
    .section-card h3 {
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Flood probability formula (based on district rainfall dataset analysis)
# Dataset: ANNUAL 95–7230 mm, mean ~1347; elevation lowers risk
# Replaces ML model trained on 1–10 scale which saturated on real values
# -------------------------------
# Rainfall-based categories requested by user
RAIN_CAT_LOW = 500.0
RAIN_CAT_MED = 1000.0

def predict_flood_probability(rainfall_mm: float, elevation_m: float) -> float:
    """Predict flood probability using explicit rainfall-category mapping and elevation.

    Categories: Low (<500 mm), Medium (500–1000 mm), High (>1000 mm).
    Returns a probability (0-1) that varies within each category and is nudged by elevation.
    """
    # Base probability by explicit rainfall ranges
    if rainfall_mm < RAIN_CAT_LOW:
        # map 0..500 -> 0.05 .. 0.29
        base = 0.05 + (rainfall_mm / RAIN_CAT_LOW) * (0.29 - 0.05)
    elif rainfall_mm <= RAIN_CAT_MED:
        # map 500..1000 -> 0.30 .. 0.69
        base = 0.30 + ((rainfall_mm - RAIN_CAT_LOW) / (RAIN_CAT_MED - RAIN_CAT_LOW)) * (0.69 - 0.30)
    else:
        # map 1000..(large) -> 0.70 .. 0.95 (asymptotic)
        frac = min(1.0, (rainfall_mm - RAIN_CAT_MED) / 3000.0)
        base = 0.70 + frac * (0.95 - 0.70)

    # Elevation effect: higher elevation reduces risk (smooth multiplier)
    elev_factor = float(np.clip(1.15 - elevation_m / 800.0, 0.6, 1.15))

    prob = float(np.clip(base * elev_factor, 0.02, 0.98))
    return round(prob, 4)

# -------------------------------
# Load district data from CSV
# -------------------------------
@st.cache_data
def load_district_data():
    df = pd.read_csv("rainfall_folder/district wise rainfall normal.csv")
    df = df.dropna(subset=["DISTRICT", "ANNUAL"])
    df["Rainfall"] = df["ANNUAL"].astype(float)
    # If real elevation data is not available, estimate elevation from rainfall
    # (heuristic): districts with very high annual rainfall are often lower-lying
    # or have larger catchments; we generate a reproducible estimate with small noise
    rain = df["Rainfall"].astype(float)
    if "Elevation" not in df.columns:
        rng = np.random.default_rng(42)
        norm = (rain - rain.min()) / (rain.max() - rain.min())
        elev_est = 800 - norm * 700 + rng.normal(0, 50, size=len(df))
        elev_est = np.clip(elev_est, 0, 2000)
        df["Elevation"] = elev_est.round(1)

    # Compute dataset rainfall bounds for percentile mapping
    global RAIN_MIN, RAIN_MAX
    RAIN_MIN = float(rain.min())
    RAIN_MAX = float(rain.max())

    # Vectorised probability using rainfall percentile and elevation factor
    rain_pct = (rain - RAIN_MIN) / (RAIN_MAX - RAIN_MIN)
    prob_rain = 0.05 + 0.90 * rain_pct
    elev_factor = np.clip(1.2 - df["Elevation"].astype(float) / 600.0, 0.5, 1.15)
    prob = np.clip(prob_rain * elev_factor, 0.02, 0.98)
    df["Flood_Probability"] = np.round(prob, 4)
    # Risk level determined from rainfall categories (user-requested thresholds)
    def risk_from_rain(r):
        if r["Rainfall"] < RAIN_CAT_LOW:
            return "Low"
        elif r["Rainfall"] <= RAIN_CAT_MED:
            return "Medium"
        else:
            return "High"

    df["Risk_Level"] = df.apply(risk_from_rain, axis=1)
    return df

@st.cache_data
def load_districts():
    df = pd.read_csv("rainfall_folder/district wise rainfall normal.csv")
    return df["DISTRICT"].dropna().unique().tolist()

district_list = load_districts()

# -------------------------------
# Helper: Causes, harmful effects, control, suggestions
# -------------------------------
def get_causes(prob: float, district: str = "", rainfall_mm: Optional[float] = None) -> str:
    """Return causes tailored by rainfall category and probability."""
    if rainfall_mm is None:
        # fallback to probability-based messaging
        if prob < 0.3:
            return "• Localised drainage/terrain factors; routine rainfall patterns."
        elif prob < 0.7:
            return "• Sustained or heavy rainfall; blocked drains and reduced conveyance capacity."
        else:
            return "• Very heavy rainfall or river/catchment saturation; possible dam or river releases."

    # Use explicit rainfall categories
    if rainfall_mm < RAIN_CAT_LOW:
        return (
            "• Low rainfall (<500 mm): localized short-duration events or convective storms.\n"
            "• Main causes: sudden intense showers, blocked local drains, minor terrain pooling."
        )
    elif rainfall_mm <= RAIN_CAT_MED:
        return (
            "• Moderate rainfall (500–1000 mm): prolonged or repeated rains that can exceed local drainage.\n"
            "• Main causes: sustained monsoon spells, inadequate culverts, agricultural runoff."
        )
    else:
        return (
            "• High rainfall (>1000 mm): catchment saturation, river overflow and large-scale runoff.\n"
            "• Main causes: heavy monsoon, upstream catchment accumulation, embankment failures, urbanisation."
        )

def get_harmful_effects(prob: float, district: str = "", rainfall_mm: Optional[float] = None) -> str:
    district_pre = f"For {district}: " if district else ""
    if rainfall_mm is None:
        # fallback to probability-only messaging
        if prob < 0.3:
            return f"{district_pre}Minor waterlogging in low spots; temporary road damage; negligible health risk."
        elif prob < 0.7:
            return f"{district_pre}Property damage in low-lying areas; crop loss; disrupted transport and power; increased disease risk."
        else:
            return f"{district_pre}Severe damage to infrastructure, risk to life, long-term displacement and major economic loss."

    # Use rainfall categories for clearer, consistent messaging
    if rainfall_mm < RAIN_CAT_LOW:
        return (
            f"{district_pre}Localized waterlogging and temporary road closures; watch for short-duration flash flooding in low spots."
        )
    elif rainfall_mm <= RAIN_CAT_MED:
        return (
            f"{district_pre}Wider waterlogging, crop stress, transport disruption and moderate property damage in vulnerable areas; heightened disease risk."
        )
    else:
        note = "Areas with very high rainfall may see more severe and widespread impacts. " if rainfall_mm > 2000 else ""
        return (
            f"{district_pre}{note}Severe flooding, infrastructure damage, potential loss of life, major crop and livestock losses, and long-term displacement."
        )

# -------------------------------
# Manual input form
# -------------------------------
st.subheader("📥 Enter Rainfall, Elevation & District (Quick Predict)")
with st.expander("Manual input – get instant prediction", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        rain_input = st.number_input("Rainfall (mm)", min_value=0.0, value=1200.0, step=50.0)
    with c2:
        elev_input = st.number_input("Elevation (m)", min_value=0.0, value=100.0, step=10.0)
    with c3:
        district_input = st.selectbox("District", options=district_list, index=min(20, len(district_list) - 1))

    if st.button("🔮 Predict Flood Risk"):
        prob = predict_flood_probability(rain_input, elev_input)
        risk = "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High"

        st.success(f"**Flood probability:** {prob:.2%} | **Risk level:** {risk}")
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Causes**")
            st.info(get_causes(prob, district_input, rain_input))
            st.markdown("**Control measures**")
            if prob < 0.3:
                st.caption("Keep drains clear; monitor rainfall and water levels.")
            elif prob < 0.7:
                st.caption("Pre-position sandbags and pumps; alert communities; deploy field teams.")
            else:
                st.caption("Evacuate vulnerable areas; close unsafe roads; deploy rescue teams and boats.")
        with col_b:
            st.markdown("**Harmful effects**")
            st.warning(get_harmful_effects(prob, district_input, rain_input))
            st.markdown("**Suggestion**")
            if prob < 0.3:
                st.caption("Routine monitoring; maintain drainage.")
            elif prob < 0.7:
                st.caption("Prepare; put teams on standby; check vulnerable locations.")
            else:
                st.caption("Activate flood response plan; warn communities; ready evacuation.")

st.markdown("---")

# -------------------------------
# District-wise analysis (from dataset)
# -------------------------------
st.subheader("📍 District-wise Flood & Rainfall Analysis")
df = load_district_data()

# Add Causes, Harmful_Effects, Suggestion
def flood_advice(prob, rainfall_mm: Optional[float] = None) -> str:
    """Return suggestion text based on the explicit rainfall categories (user-provided thresholds)."""
    if rainfall_mm is None:
        # fallback to probability-driven advice
        if prob < 0.3:
            return "Situation: Normal. Keep routine monitoring and maintain drains."
        elif prob < 0.7:
            return "Situation: Alert. Prepare teams, pre-position pumps and sandbags, and check vulnerable spots."
        else:
            return "Situation: Emergency. Activate response plan, warn communities and ready evacuation."

    if rainfall_mm < RAIN_CAT_LOW:
        return "Routine monitoring; clear local drains; advise communities to avoid low-lying spots during heavy showers."
    elif rainfall_mm <= RAIN_CAT_MED:
        return "Prepare: put teams on standby, inspect drains and culverts, pre-position sandbags and pumps, and alert vulnerable communities."
    else:
        advice = "Emergency: activate flood response plan, warn communities, and prepare evacuations."
        if rainfall_mm > 1500:
            advice += " Coordinate with upstream authorities, check embankments and mobilise extra rescue resources."
        return advice

# Populate causes/effects/suggestions using rainfall-aware helpers
df["Causes"] = df.apply(lambda r: get_causes(r["Flood_Probability"], r.get("DISTRICT", ""), r.get("Rainfall", None)), axis=1)
df["Harmful_Effects"] = df.apply(lambda r: get_harmful_effects(r["Flood_Probability"], r.get("DISTRICT", ""), r.get("Rainfall", None)), axis=1)
df["Suggestion"] = df.apply(lambda r: flood_advice(r["Flood_Probability"], r.get("Rainfall", None)), axis=1)

# Overview metrics
total = len(df)
avg_prob = float(df["Flood_Probability"].mean())
high_risk = int((df["Risk_Level"] == "High").sum())

with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total districts", f"{total}")
    col2.metric("Average flood probability", f"{avg_prob:.2%}")
    col3.metric("High-risk districts", f"{high_risk}")
    st.markdown('</div>', unsafe_allow_html=True)

if high_risk > 0:
    st.warning(f"⚠️ {high_risk} districts are at HIGH risk!")

# District summary table
st.subheader("District-wise Summary")
display_cols = ["STATE_UT_NAME", "DISTRICT", "Rainfall", "Flood_Probability", "Risk_Level", "Causes", "Harmful_Effects", "Suggestion"]
display_cols = [c for c in display_cols if c in df.columns]
st.dataframe(df[display_cols], use_container_width=True, height=400)

# Charts
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Flood Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Flood_Probability"], bins=20, color='#38bdf8', edgecolor='black')
    ax.set_xlabel("Flood Probability")
    ax.set_ylabel("Number of districts")
    st.pyplot(fig)

    st.subheader("Risk Level Distribution")
    fig2, ax2 = plt.subplots()
    counts = df["Risk_Level"].value_counts()
    order = [c for c in ["Low", "Medium", "High"] if c in counts.index]
    colors = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"}
    counts.reindex(order, fill_value=0).plot(kind="bar", ax=ax2, color=[colors.get(c, "#94a3b8") for c in order])
    ax2.set_xlabel("Risk Level")
    ax2.set_ylabel("Number of districts")
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# Filter by risk
st.subheader("Filter by Risk Level")
selected_risk = st.multiselect(
    "Select Risk Level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"]
)
filtered = df[df["Risk_Level"].isin(selected_risk)]
st.dataframe(filtered[display_cols], use_container_width=True, height=300)

# Download
st.download_button(
    label="Download District-wise Summary as CSV",
    data=df[display_cols].to_csv(index=False).encode("utf-8"),
    file_name="district_flood_rainfall_summary.csv",
    mime="text/csv",
)
