import requests
import pandas as pd

def fetch_nasa_weather(lat, lon):
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M,RH2M,PRECTOT",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": "20150101",
        "end": "20241231",
        "format": "JSON"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Extract values
    records = data["properties"]["parameter"]
    print(records.keys())

    df = pd.DataFrame({
        "date": records["T2M"].keys(),
        "temperature": records["T2M"].values(),
        "humidity": records["RH2M"].values(),
        "rainfall": records["PRECTOTCORR"].values()
    })

    df = df.dropna()

    return df

df_nasa = fetch_nasa_weather(11.4064, 76.6932)

print(df_nasa.head())

df_nasa["rainfall"] = df_nasa["rainfall"] * 365

df_nasa.to_csv("data/nasa_weather.csv", index=False)