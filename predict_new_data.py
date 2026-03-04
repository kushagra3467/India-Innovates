import pandas as pd
import joblib
import os

# =====================================================
# CREATE SAMPLE NEW DATA IF MISSING
# =====================================================
if not os.path.exists("new_rain_elev_data.xlsx"):
    data = {
        "Date": pd.date_range("2026-03-11", periods=10),
        "Rainfall": [10,20,30,25,15,5,35,40,50,45],
        "Elevation": [100,110,120,130,140,150,160,170,180,190]
    }
    df = pd.DataFrame(data)
    df.to_excel("new_rain_elev_data.xlsx", index=False)
    print("Sample new_rain_elev_data.xlsx created!")

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
model = joblib.load("flood_model.pkl")
scaler = joblib.load("scaler.pkl")

# =====================================================
# LOAD NEW DATA
# =====================================================
new_data = pd.read_excel("new_rain_elev_data.xlsx")

# =====================================================
# ENSURE ONLY NUMERIC FEATURES ARE USED
# =====================================================
feature_cols = ["Rainfall", "Elevation"]  # MUST match training features exactly
X_new = new_data[feature_cols]
X_new_scaled = scaler.transform(X_new)

# =====================================================
# PREDICTION
# =====================================================
new_data["Flood_Probability"] = model.predict_proba(X_new_scaled)[:, 1]
new_data["Risk_Level"] = new_data["Flood_Probability"].apply(
    lambda p: "Low" if p < 0.3 else "Medium" if p < 0.7 else "High"
)

# =====================================================
# SAVE NEW PREDICTIONS
# =====================================================
new_data.to_csv("new_flood_predictions.csv", index=False)
print("Predictions saved in new_flood_predictions.csv ✅")
print(new_data)

# =========================
# PLOTS (Add this here)
# =========================
import matplotlib.pyplot as plt

# Flood probability histogram
plt.hist(new_data["Flood_Probability"], bins=10)
plt.title("Flood Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()

# Risk level count
new_data["Risk_Level"].value_counts().plot(kind="bar", title="Risk Level Distribution")
plt.show()