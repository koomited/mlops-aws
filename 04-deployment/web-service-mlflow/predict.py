import pickle
from flask import Flask, request, jsonify
import mlflow

MLFLOW_TRACKING_URI = "http://localhost:5000"
RUN_ID = "5d820d847ab242aba000ea2b5877c7f8"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("green-taxi-duration")


model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(model_uri)


    
def prepare_features(ride):
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    
    return features
    
def predict(features):
    preds = model.predict(features)
    return  preds


app = Flask("duration-prediction")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    y_pred = predict(features)
    result = {
        "duration": float(y_pred[0])
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)