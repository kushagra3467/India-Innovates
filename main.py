import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# =====================================================
# 0️⃣ CREATE SAMPLE FOLDERS & EXCEL FILES IF MISSING
# =====================================================
def create_sample_excel(folder, filename, columns):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        # create sample DataFrame with 10 rows
        data = {col: range(1, 11) for col in columns}
        df = pd.DataFrame(data)
        df.to_excel(path, index=False)
        print(f"Sample Excel created: {path}")
    return path

# Create sample rainfall and elevation folders & Excel
rain_folder = "rainfall_folder"
elev_folder = "elevation_folder"

create_sample_excel(rain_folder, "rain1.xlsx", ["Date", "Rainfall"])
create_sample_excel(elev_folder, "elev1.xlsx", ["Date", "Elevation"])

# =====================================================
# FUNCTION: SAFE LOAD EXCEL FILES
# =====================================================
def load_excel_folder(folder_path):
    files = glob.glob(f"{folder_path}/*.xlsx")
    print(f"Files found in {folder_path}: {files}")
    df_list = []
    for file in files:
        try:
            df = pd.read_excel(file)
            if not df.empty:
                df_list.append(df)
            else:
                print(f"Warning: {file} is empty, skipped.")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        print(f"No valid Excel files found in {folder_path}. Returning empty DataFrame.")
        return pd.DataFrame()

# =====================================================
# 1️⃣ LOAD RAINFALL & ELEVATION DATA
# =====================================================
rain_df = load_excel_folder(rain_folder)
elev_df = load_excel_folder(elev_folder)

if rain_df.empty or elev_df.empty:
    raise ValueError("Rainfall or Elevation data is empty. Please check your Excel files!")

print("Rain DF shape:", rain_df.shape)
print("Elevation DF shape:", elev_df.shape)

# =====================================================
# 2️⃣ MERGE DATA ON DATE
# =====================================================
if "Date" not in rain_df.columns or "Date" not in elev_df.columns:
    raise KeyError("Column 'Date' missing in rainfall or elevation data!")

final_df = pd.merge(rain_df, elev_df, on="Date", how="inner")
print("Final DF shape after merge:", final_df.shape)

# =====================================================
# 3️⃣ HANDLE MISSING VALUES
# =====================================================
print("Missing values before drop:", final_df.isnull().sum())
final_df = final_df.dropna()
print("Final DF shape after dropping missing values:", final_df.shape)

# =====================================================
# 4️⃣ ADD SAMPLE TARGET 'Flood' IF MISSING
# =====================================================
if "Flood" not in final_df.columns:
    # create sample binary target
    final_df["Flood"] = [0,1]* (len(final_df)//2) + [0]*(len(final_df)%2)

# =====================================================
# 5️⃣ DEFINE FEATURES & TARGET
# =====================================================
X = final_df.drop(columns=["Flood","Date"])
y = final_df["Flood"]

# =====================================================
# 6️⃣ TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 7️⃣ SCALING
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 8️⃣ MODEL TRAINING
# =====================================================
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# =====================================================
# 9️⃣ EVALUATION
# =====================================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =====================================================
# 🔟 CONFUSION MATRIX
# =====================================================
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================================
# 1️⃣1️⃣ ROC CURVE & AUC
# =====================================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
print("AUC Score:", auc_score)

# =====================================================
# 1️⃣2️⃣ FEATURE IMPORTANCE
# =====================================================
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\nFeature Importance:\n", feature_df)

plt.figure(figsize=(8,6))
plt.barh(feature_df["Feature"], feature_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.show()

# =====================================================
# 1️⃣3️⃣ SAVE MODEL & SCALER
# =====================================================
joblib.dump(model, "flood_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel & Scaler Saved ✅")

# =====================================================
# 1️⃣4️⃣ FLOOD PROBABILITY ON FULL DATA
# =====================================================
X_scaled_full = scaler.transform(X)
final_df["Flood_Probability"] = model.predict_proba(X_scaled_full)[:, 1]

# =====================================================
# 1️⃣5️⃣ RISK CATEGORY
# =====================================================
def risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

final_df["Risk_Level"] = final_df["Flood_Probability"].apply(risk_level)

# =====================================================
# 1️⃣6️⃣ PROBABILITY DISTRIBUTION
# =====================================================
plt.figure()
plt.hist(final_df["Flood_Probability"], bins=20, color="skyblue", edgecolor="black")
plt.title("Flood Probability Distribution")
plt.xlabel("Flood Probability")
plt.ylabel("Count")
plt.show()

# =====================================================
# 1️⃣7️⃣ RISK LEVEL DISTRIBUTION
# =====================================================
plt.figure()
final_df["Risk_Level"].value_counts().plot(kind="bar", color="orange")
plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.show()

# =====================================================
# 1️⃣8️⃣ WARD READINESS SCORE (optional)
# =====================================================
if "Ward" in final_df.columns:
    ward_score = final_df.groupby("Ward")["Flood_Probability"].mean().reset_index()
    ward_score["Readiness_Score"] = (1 - ward_score["Flood_Probability"]) * 100
    ward_score.to_csv("ward_readiness_score.csv", index=False)
    print("\nWard Readiness Score:\n", ward_score.head())

# =====================================================
# 1️⃣9️⃣ SAVE FINAL OUTPUT
# =====================================================
final_df.to_csv("final_flood_prediction_output.csv", index=False)
print("\nFiles Saved Successfully ✅")
print("\n🚀 AI FLOOD PREDICTION SYSTEM COMPLETED SUCCESSFULLY 🚀")