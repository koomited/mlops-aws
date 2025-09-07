import pickle
from flask import Flask, request, jsonify

with open("ridge_reg.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)
    
def prepare_features(ride):
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    
    return features
    
def predict(features):
    X = dv.transform(features)
    y_pred = model.predict(X)
    return y_pred

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