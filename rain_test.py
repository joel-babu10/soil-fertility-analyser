import requests

API_KEY = "3c466ca4c6d8e0a29788218f2dc189b2"
CITY = "KOTTAYAM"   # change to your city

url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

# Rainfall may not always exist
rainfall = data.get("rain", {}).get("1h", 0)

print("Rainfall (last 1 hour):", rainfall, "mm")
