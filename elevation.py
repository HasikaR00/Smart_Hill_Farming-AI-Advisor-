import requests
def get_altitude(lat, lon):

    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"

    response = requests.get(url)
    data = response.json()

    return data["results"][0]["elevation"]