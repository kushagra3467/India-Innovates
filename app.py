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
            return "Low risk – Normal monitoring"
        elif prob < 0.7:
            return "Medium risk – Prepare sandbags and review drainage"
        else:
            return "High risk – Evacuation plan, deploy emergency teams"

    # -------------------------------
    # Mitigation / Control Suggestions
    # -------------------------------
    def mitigation_tips(prob):
        if prob < 0.3:
            return "Keep drains clear, monitor rainfall"
        elif prob < 0.7:
            return "Temporary sandbags, inspect drainage, alert community"
        else:
            return "Evacuate low-lying areas, deploy pumps, alert hospitals, restrict traffic"

    new_data["Suggestion"] = new_data["Flood_Probability"].apply(flood_advice)
    new_data["Control_Tips"] = new_data["Flood_Probability"].apply(mitigation_tips)

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
    st.subheader("Recommended Actions & Control Measures")
    display_cols = ["Ward","Flood_Probability","Risk_Level","Suggestion","Control_Tips"]
    display_cols = [col for col in display_cols if col in new_data.columns]
    st.dataframe(new_data[display_cols])

    # -------------------------------
    # High Risk Alert
    # -------------------------------
    high_risk_count = (new_data["Flood_Probability"] > 0.7).sum()
    if high_risk_count > 0:
        st.warning(f"⚠️ {high_risk_count} areas are at HIGH risk!")

    # -------------------------------
    # Charts
    # -------------------------------
    st.subheader("Flood Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(new_data["Flood_Probability"], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel("Flood Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Risk Level Distribution")
    fig2, ax2 = plt.subplots()
    new_data["Risk_Level"].value_counts().plot(kind="bar", ax=ax2, color='salmon')
    ax2.set_xlabel("Risk Level")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

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