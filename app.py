# app.py
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import joblib
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
# Load model & scaler
# -------------------------------
model = joblib.load("flood_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Upload new data
# -------------------------------
st.subheader("Upload Rainfall & Elevation Data (Excel)")
uploaded_file = st.file_uploader("Choose Excel file", type="xlsx")

if uploaded_file:
    new_data = pd.read_excel(uploaded_file)

    # -------------------------------
    # Ensure numeric features only
    # -------------------------------
    feature_cols = ["Rainfall", "Elevation"]  # Must match training features
    X_new = new_data[feature_cols]
    X_new_scaled = scaler.transform(X_new)

    # -------------------------------
    # Predict Flood Probability & Risk
    # -------------------------------
    new_data["Flood_Probability"] = model.predict_proba(X_new_scaled)[:, 1]
    new_data["Risk_Level"] = new_data["Flood_Probability"].apply(
        lambda p: "Low" if p < 0.3 else "Medium" if p < 0.7 else "High"
    )

    # -------------------------------
    # Recommended Actions
    # -------------------------------
    def flood_advice(prob):
        if prob < 0.3:
            return "Situation: Normal.\nAction: Keep routine monitoring and maintain drainage; no special measures needed."
        elif prob < 0.7:
            return "Situation: Alert.\nAction: Put teams on standby, check vulnerable locations, and prepare sandbags and pumps."
        else:
            return "Situation: Emergency likely.\nAction: Activate flood response plan, warn communities, and be ready to evacuate."

    # -------------------------------
    # Mitigation / Control Suggestions
    # -------------------------------
    def mitigation_tips(prob):
        if prob < 0.3:
            return (
                "Before flood: keep drains and culverts clear, protect natural water channels.\n"
                "During heavy rain: monitor water levels at low spots, log any waterlogging.\n"
                "After events: inspect and repair minor damages to roads and embankments."
            )
        elif prob < 0.7:
            return (
                "Before flood: pre-position sandbags and mobile pumps at known hotspots, test sirens and communication channels.\n"
                "During heavy rain: deploy field teams to monitor critical locations, issue local alerts to at‑risk communities.\n"
                "After flood: assess damage, clear debris from drains, and update risk maps using observed flood extents."
            )
        else:
            return (
                "Before flood: confirm evacuation routes and shelters, move vulnerable people (elderly, hospitals) out of high‑risk zones.\n"
                "During flood: evacuate low‑lying areas, close unsafe roads, deploy rescue teams, pumps and boats; keep hospitals and power stations protected.\n"
                "After flood: run health camps, restore drinking water and sanitation, and plan medium‑term works (embankments, retention ponds, wetland restoration)."
            )

    new_data["Suggestion"] = new_data["Flood_Probability"].apply(flood_advice)
    new_data["Control_Tips"] = new_data["Flood_Probability"].apply(mitigation_tips)

    # -------------------------------
    # At-a-glance overview metrics
    # -------------------------------
    total_areas = len(new_data)
    avg_prob = float(new_data["Flood_Probability"].mean())
    high_risk_count = int((new_data["Risk_Level"] == "High").sum())

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total records", f"{total_areas}")
        col2.metric("Average flood probability", f"{avg_prob:.2f}")
        col3.metric("High-risk locations", f"{high_risk_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Helper: detect district column
    # -------------------------------
    def get_district_column(df: pd.DataFrame) -> Optional[str]:
        possible_names = ["DISTRICT", "District", "district"]
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    district_col = get_district_column(new_data)

    # -------------------------------
    # Display Recommendations & Controls
    # -------------------------------
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Recommended Actions & Control Measures")
        display_cols = ["Ward","Flood_Probability","Risk_Level","Suggestion","Control_Tips"]
        display_cols = [col for col in display_cols if col in new_data.columns]
        st.dataframe(new_data[display_cols])
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # High Risk Alert
    # -------------------------------
    if high_risk_count > 0:
        st.warning(f"⚠️ {high_risk_count} areas are at HIGH risk!")

    # -------------------------------
    # Charts
    # -------------------------------
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Flood Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(new_data["Flood_Probability"], bins=10, color='#38bdf8', edgecolor='black')
        ax.set_xlabel("Flood Probability")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Risk Level Distribution")
        fig2, ax2 = plt.subplots()
        new_data["Risk_Level"].value_counts().plot(kind="bar", ax=ax2, color='#fb7185')
        ax2.set_xlabel("Risk Level")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Ward Readiness Score
    # -------------------------------
    if "Ward" in new_data.columns:
        ward_score = new_data.groupby("Ward")["Flood_Probability"].mean().reset_index()
        ward_score["Readiness_Score"] = (1 - ward_score["Flood_Probability"]) * 100
        st.subheader("Ward-wise Readiness Score")
        st.dataframe(ward_score)

    # -------------------------------
    # District-wise Rainfall & Flood Summary
    # -------------------------------
    if district_col is not None:
        st.subheader("District-wise Rainfall & Flood Summary")

        agg_dict = {
            "Flood_Probability": ["mean", "max"]
        }
        if "Rainfall" in new_data.columns:
            agg_dict["Rainfall"] = ["mean", "max"]

        district_summary = (
            new_data.groupby(district_col)
            .agg(agg_dict)
        )

        # Flatten MultiIndex columns
        district_summary.columns = [
            "_".join([c for c in col if c]).strip("_")
            for col in district_summary.columns.to_flat_index()
        ]
        district_summary = district_summary.reset_index()

        if "Risk_Level" in new_data.columns:
            mode_risk = (
                new_data.groupby(district_col)["Risk_Level"]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
                .reset_index()
                .rename(columns={"Risk_Level": "Dominant_Risk_Level"})
            )
            district_summary = district_summary.merge(mode_risk, on=district_col, how="left")

            # Add district-level suggestion text based on dominant risk
            def district_suggestion(level: str) -> str:
                if level == "Low":
                    return (
                        "Overall situation: generally safe.\n"
                        "District actions: keep drains and natural channels clear, maintain gauges and warning systems, "
                        "and regularly review local low‑lying pockets for waterlogging."
                    )
                if level == "Medium":
                    return (
                        "Overall situation: watch and prepare.\n"
                        "District actions: desilt major drains before monsoon, pre‑position sandbags and pumps at hotspots, "
                        "conduct community awareness drives and mock drills in flood‑prone villages/wards."
                    )
                if level == "High":
                    return (
                        "Overall situation: high flood risk.\n"
                        "District actions: finalize and test evacuation plans, protect critical infrastructure (hospitals, power, water works), "
                        "identify safe shelters and stock them, and enforce controls on new construction in flood‑prone areas."
                    )
                return (
                    "Overall situation: unclear from data.\n"
                    "District actions: verify local flood history, check gauge and rainfall data quality, and update exposure maps."
                )

            district_summary["District_Suggestion"] = district_summary["Dominant_Risk_Level"].apply(district_suggestion)

        st.dataframe(district_summary)

        st.download_button(
            label="Download District-wise Summary as CSV",
            data=district_summary.to_csv(index=False).encode("utf-8"),
            file_name="district_flood_rainfall_summary.csv",
            mime="text/csv",
        )

    # -------------------------------
    # Filter by Risk Level
    # -------------------------------
    st.subheader("Filter Data by Risk Level")
    selected_risk = st.multiselect(
        "Select Risk Level", ["Low","Medium","High"], default=["Low","Medium","High"]
    )
    filtered_data = new_data[new_data["Risk_Level"].isin(selected_risk)]
    st.dataframe(filtered_data)

    # -------------------------------
    # Download Predictions
    # -------------------------------
    st.download_button(
        label="Download Predictions as CSV",
        data=new_data.to_csv(index=False).encode('utf-8'),
        file_name="flood_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload an Excel file to see predictions and control suggestions.")