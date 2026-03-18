import os
import sys
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.weather import fetch_current_weather
from api.elevation import get_altitude

load_dotenv()

# ----------------------------
# LOAD MODELS
# ----------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..')
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
crop_names = joblib.load(os.path.join(MODELS_DIR, "crop_names.pkl"))
planting_calendar = joblib.load(os.path.join(MODELS_DIR, "planting_calendar.pkl"))
agri_schedule = joblib.load(os.path.join(MODELS_DIR, "agri_schedule.pkl"))

# ----------------------------
# FEATURE PREP
# ----------------------------
def prepare_features(weather, soil, irrigation_days, altitude=None):
    features = {
        "temperature": weather.get("temperature", 0),
        "humidity": weather.get("humidity", 0),
        "rainfall": weather.get("rainfall", 0),
        "soil_ph": soil.get("soil_ph", 0),
        "nitrogen": soil.get("nitrogen", 0),
        "phosphorus": soil.get("phosphorus", 0),
        "potassium": soil.get("potassium", 0),
        "irrigation_days": irrigation_days,
    }

    if altitude:
        features["altitude"] = altitude

    df = pd.DataFrame([features])

    for col in scaler.feature_names_in_:
        if col not in df:
            df[col] = 0

    df = df[scaler.feature_names_in_]
    return scaler.transform(df)

# ----------------------------
# WEATHER TREND
# ----------------------------
def analyze_trends(weather):
    return {
        "rain": "increasing" if weather.get("rainfall", 0) > 10 else "low",
        "temp": "high" if weather.get("temperature", 0) > 30 else "normal"
    }

# ----------------------------
# SMART IRRIGATION
# ----------------------------
def dynamic_irrigation(weather):
    rain = weather.get("rainfall", 0)

    if rain < 5:
        return 3
    elif rain < 15:
        return 5
    else:
        return 10

# ----------------------------
# EXPLAINABILITY
# ----------------------------
def generate_explanation(weather, soil):
    reasons = []
    temp = weather.get("temperature", 0)

    if 15 <= temp <= 25:
        reasons.append("🌡️ Optimal temperature")
    elif temp < 15:
        reasons.append("❄️ Cool temperature suitable for hill crops")

    if weather.get("humidity", 0) > 60:
        reasons.append("💧 Good humidity")

    if 6 <= soil.get("soil_ph", 0) <= 7:
        reasons.append("🧪 Ideal soil pH")

    if soil.get("nitrogen", 0) > 200:
        reasons.append("🌱 High nitrogen")

    return reasons

# ----------------------------
# CONFIDENCE LABEL
# ----------------------------
def get_confidence_label(prob):
    if prob >= 85:
        return "High"
    elif prob >= 70:
        return "Medium"
    else:
        return "Low"

# ----------------------------
# ALERT ENGINE
# ----------------------------
def generate_alerts(weather, trends):
    alerts = []

    if weather.get("humidity", 0) > 80:
        alerts.append("⚠️ Fungal risk high")

    if weather.get("rainfall", 0) > 20:
        alerts.append("🌧️ Heavy rain → Delay sowing")

    if trends["rain"] == "low":
        alerts.append("☀️ Dry period → Good for sowing")

    if trends["temp"] == "high":
        alerts.append("🔥 High temperature → Increase irrigation")

    return alerts

# ----------------------------
# CROP ROTATION
# ----------------------------
def generate_crop_rotation(top_crops):
    if len(top_crops) < 3:
        return "Not enough crops for rotation"

    return f"""
Season 1: {top_crops[0]['crop']}
Season 2: {top_crops[1]['crop']}
Season 3: {top_crops[2]['crop']}
Then rotate with legumes for soil recovery
"""

# ----------------------------
# 🤖 OPENROUTER LLM (FREE)
# ----------------------------
def get_llm_advice(prompt):
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            return None

        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                # ✅ FREE MODEL
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400
            },
            timeout=10
        )

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print("LLM Error:", e)

    return None

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def predict_crops(lat, lon, soil_card, irrigation_yes=True, top_n=3):

    weather = fetch_current_weather(lat, lon)
    altitude = get_altitude(lat, lon)

    trends = analyze_trends(weather)
    irrigation_days = dynamic_irrigation(weather)

    crop_scores = []

    for crop in crop_names:
        features = prepare_features(weather, soil_card, irrigation_days, altitude)

        prob = float(model.predict_proba(features)[0][1] * 100)

        crop_scores.append({
            "crop": crop,
            "probability": round(prob, 2),
            "confidence": get_confidence_label(prob),
            "planting_dates": planting_calendar.get(crop, ["No data available"]),
            "irrigation_days": irrigation_days,
            "fertilizer": agri_schedule.get(crop, {}).get("fertilizer", {}),
            "reasons": generate_explanation(weather, soil_card)
        })

    # Ranking
    top_crops = sorted(crop_scores, key=lambda x: x["probability"], reverse=True)[:top_n]

    for i, crop in enumerate(top_crops):
        crop["rank"] = i + 1

    alerts = generate_alerts(weather, trends)

    # Prompt
    crop_info = "\n".join([f"{c['crop']} ({c['probability']}%)" for c in top_crops])

    prompt = f"""
You are an agricultural AI.

Location: {lat},{lon}
Soil: {soil_card}
Weather: {weather}

Top crops:
{crop_info}

Alerts:
{alerts}

Provide:
- Crop rotation
- Irrigation advice
- Fertilizer strategy
- Risk mitigation
"""

    advice = get_llm_advice(prompt)

    # 🛡️ RULE-BASED FALLBACK
    if advice is None:
        advice = f"""
🌱 Crop Rotation:
{top_crops[0]['crop']} → {top_crops[1]['crop']} → {top_crops[2]['crop']}

💧 Irrigation:
Every {irrigation_days} days

🧪 Fertilizer:
{top_crops[0]['fertilizer']}

⚠️ Alerts:
{', '.join(alerts) if alerts else "Low risk"}
"""

    return {
        "weather": weather,
        "altitude": altitude,
        "top_crops": top_crops,
        "rotation_plan": generate_crop_rotation(top_crops),
        "alerts": alerts,
        "ai_advice": advice
    }

# ----------------------------
# TEST
# ----------------------------
if __name__ == "__main__":
    soil = {
        "soil_ph": 6.5,
        "nitrogen": 300,
        "phosphorus": 50,
        "potassium": 200
    }

    result = predict_crops(11.4064, 76.6932, soil)

    import json

    print(json.dumps(result, indent=4))