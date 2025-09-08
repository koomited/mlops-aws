import model



def test_base64_decode():
    base64_input = "eyJyaWRlIjogeyJQVUxvY2F0aW9uSUQiOiAxMzAsICJET0xvY2F0aW9uSUQiOiAyMDUsICJ0cmlwX2Rpc3RhbmNlIjogMy43NX0sICJyaWRlX2lkIjogMjU2fQ=="
    actual_result = model.base64_decode(base64_input)
    
    expected_result = {
                "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.75
            },
            "ride_id": 256
    }
    
    assert actual_result == expected_result
    
def test_prepare_features():
    model_service = model.ModelService(None)
    ride = {
         "PULocationID": 130,
      "DOLocationID": 205,
      "trip_distance": 3.75
    }
    
    actual_features = model_service.prepare_features(ride)
    
    expected_features = {
        "PU_DO":"130_205",
        "trip_distance":3.75
    }
    
    assert actual_features == expected_features
    
    
class ModelMock:
    def __init__(self, value) -> None:
        self.value = value
        
    
    def predict(self, X):
        n = len(X)
        return [10.0]*n

def test_predict():
    
    model_mock = ModelMock(10)
    model_service = model.ModelService(model_mock)
    features = {
        "PU_DO":"130_205",
        "trip_distance":3.75
    }

    actual_prediction = model_service.predict(features)
    expected_prediction = 10
    
    assert actual_prediction == expected_prediction
    
    
def test_lambda_handler():
    model_version = "Test123"
    model_mock = ModelMock(10)
    
    model_service = model.ModelService(model_mock, model_version)
        
    event = {
        "Records": [
            {
                "kinesis": {
                    "data": "eyJyaWRlIjogeyJQVUxvY2F0aW9uSUQiOiAxMzAsICJET0xvY2F0aW9uSUQiOiAyMDUsICJ0cmlwX2Rpc3RhbmNlIjogMy43NX0sICJyaWRlX2lkIjogMjU2fQ==",
                },
            }
              
        ]
    }

    actual_prediction = model_service.lambda_handler(event)
    expected_prediction = {
        "predictions":[{
                    "model": "ride-duration-prediction-model",
                    "version": model_version,
                    "prediction": {
                    "ride_duration":10.0,
                    "ride_id":256
                }
        }]
    }
    
    assert actual_prediction == expected_prediction
    