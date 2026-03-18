from dotenv import load_dotenv
import os
import requests
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
def fetch_current_weather(lat, lon):

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()
    print(data)

    weather = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": data.get("rain", {}).get("1h", 0)
    }

    return weather

current_weather = fetch_current_weather(11.4064, 76.6932)

print(current_weather)