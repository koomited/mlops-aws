import json
import base64
import boto3
import os
import mlflow

import mlflow

RUN_ID = "m-73b1fea3e7c0444ebff7192f9d16ed53"
model_uri = f"s3://koomi-mlflow-artifacts-remote/2/models/{RUN_ID}/artifacts"
model = mlflow.pyfunc.load_model(model_uri)

TEST_RUN = os.getenv("TEST_RUN", "false").lower() == "true"

PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME", "ride_predictions")


kinesis_client = boto3.client(
    "kinesis",
    region_name="us-east-1"  # replace with your stream's region
)



def prepare_features(ride):
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    
    return features

def predict(features):
    return  model.predict(features)[0]

def lambda_handler(event, context):
    
    # print(json.dumps(event))
    predictions_events = []
    for record in event["Records"]:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)
        # print(ride_event)

        ride = ride_event["ride"]
        ride_id = ride_event["ride_id"]
        features = prepare_features(ride)
        prediction = predict(features)
        prediction_event =  {
                "model": "ride-duration-prediction-model",
                "version": "1.0.0",
                "prediction": {
                "ride_duration":prediction,
                "ride_id":ride_id
                }
            }
        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id)
            )
        
        predictions_events.append(prediction_event)

    return {
        "predictions":predictions_events
    }
