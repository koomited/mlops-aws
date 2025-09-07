import requests


ride =  {
    "PULocationID": 130,
    "DOLocationID": 205,
    "trip_distance": 3.75
}

url = "http://localhost:9696/predict"

pred = requests.post(url, json=ride).json()

print(f"Predicted duration: {pred["duration"]:.2f} minutes")