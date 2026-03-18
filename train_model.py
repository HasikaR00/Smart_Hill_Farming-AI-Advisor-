import pandas as pd
import joblib
import os
import pathlib
import numpy as np
import requests

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------
# PROJECT ROOT
# ---------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "data" / "crop_dataset_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("📂 Loading dataset from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# ---------------------------
# FEATURES
# ---------------------------
X = df.drop(["crop", "label"], axis=1)
y = df["label"]
crop_names = df["crop"].unique().tolist()

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# SCALER
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# MODELS
# ---------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss', random_state=42)
}

best_model = None
best_score = 0
results = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# TRAIN LOOP
# ---------------------------
for name, model in models.items():
    print(f"\n🔹 Training {name}")
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)
    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=cv).mean()

    results[name] = {"acc": acc, "roc": roc, "cv": cv_score}

    if (cv_score + roc) > best_score:
        best_score = cv_score + roc
        best_model = model
        best_model_name = name

# ---------------------------
# 🌍 NASA WEATHER FETCH
# ---------------------------
def fetch_nasa_weather(lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": "20150101",
        "end": "20241231",
        "format": "JSON"
    }

    res = requests.get(url, params=params)
    data = res.json()["properties"]["parameter"]

    df = pd.DataFrame({
        "date": data["T2M"].keys(),
        "temperature": data["T2M"].values(),
        "humidity": data["RH2M"].values(),
        "rainfall": data["PRECTOTCORR"].values()
    }).dropna()

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    return df

# 📍 Default location
df_nasa = fetch_nasa_weather(11.4064, 76.6932)

# ---------------------------
# 🌱 LEARN CROP REQUIREMENTS (DATA-DRIVEN)
# ---------------------------
crop_requirements = {}

for crop in crop_names:
    crop_df = df[df["crop"] == crop]

    crop_requirements[crop] = {
        "temp": (
            crop_df["temperature"].quantile(0.25),
            crop_df["temperature"].quantile(0.75)
        ),
        "rain": (
            crop_df["rainfall"].quantile(0.25),
            crop_df["rainfall"].quantile(0.75)
        )
    }

# ---------------------------
# 📅 BEST 5-DAY WINDOWS (STARTUP LEVEL)
# ---------------------------
# ---------------------------
# 📅 SMART WINDOW ENGINE (RANKED)
# ---------------------------
def score_window(avg_temp, total_rain, crop_req):
    temp_mid = (crop_req["temp"][0] + crop_req["temp"][1]) / 2
    rain_mid = (crop_req["rain"][0] + crop_req["rain"][1]) / 2

    temp_score = 1 - abs(avg_temp - temp_mid) / (temp_mid + 1)
    rain_score = 1 - abs(total_rain - rain_mid) / (rain_mid + 1)

    return temp_score + rain_score


def format_date_range(start, end):
    return f"{start.strftime('%b %d')} – {end.strftime('%b %d')}"


def get_best_windows(df_nasa, crop_req):
    windows = []
    df_sorted = df_nasa.sort_values("date")

    for i in range(len(df_sorted) - 7):
        window = df_sorted.iloc[i:i+7]

        avg_temp = window["temperature"].mean()
        total_rain = window["rainfall"].sum()
        score = score_window(avg_temp, total_rain, crop_req)

        start = window.iloc[0]["date"]
        end = window.iloc[-1]["date"]

        windows.append((score, format_date_range(start, end)))

    # ❗ Sort by best score
    windows = sorted(windows, key=lambda x: x[0], reverse=True)

    # ✅ Return top 3 clean windows
    return [w[1] for w in windows[:3]]
# ---------------------------
# 🌱 PLANTING CALENDAR
# ---------------------------
# ---------------------------
# 🌱 PLANTING CALENDAR
# ---------------------------
print("\n🌱 Generating AI planting windows (ranked)...")

planting_calendar = {}

for crop in crop_names:
    windows = get_best_windows(df_nasa, crop_requirements[crop])

    # ❗ Fallback if no windows found
    if not windows:
        windows = ["Season not ideal currently"]

    planting_calendar[crop] = windows

# ---------------------------
# ⚠️ RISK ENGINE
# ---------------------------
def detect_risk(df):
    risks = []

    if df["humidity"].mean() > 80:
        risks.append("Fungal disease risk")

    if df["rainfall"].sum() > 2500:
        risks.append("Flood risk")

    if df["temperature"].mean() > 30:
        risks.append("Heat stress risk")

    return risks

# ---------------------------
# 💧 SMART IRRIGATION
# ---------------------------
def smart_irrigation(df):
    recent_rain = df.tail(7)["rainfall"].sum()

    if recent_rain < 10:
        return 3
    elif recent_rain < 30:
        return 5
    else:
        return 10

# ---------------------------
# 🧪 DATA-DRIVEN AGRO SCHEDULE
# ---------------------------
agri_schedule = {}

for crop in crop_names:

    crop_df = df[df["crop"] == crop]

    agri_schedule[crop] = {
        "irrigation_days": smart_irrigation(df_nasa),
        "fertilizer": {
            "N": int(crop_df["nitrogen"].mean()),
            "P": int(crop_df["phosphorus"].mean()),
            "K": int(crop_df["potassium"].mean())
        },
        "risk_flags": detect_risk(df_nasa)
    }

# ---------------------------
# SAVE
# ---------------------------
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(crop_names, MODELS_DIR / "crop_names.pkl")
joblib.dump(planting_calendar, MODELS_DIR / "planting_calendar.pkl")
joblib.dump(agri_schedule, MODELS_DIR / "agri_schedule.pkl")

print("\n🏆 Best Model:", best_model_name)
print("✅ All artifacts saved in:", MODELS_DIR)

print("\n📊 Model Comparison:")
for k, v in results.items():
    print(f"{k}: Acc={v['acc']:.3f}, CV={v['cv']:.3f}, ROC={v['roc']:.3f}")