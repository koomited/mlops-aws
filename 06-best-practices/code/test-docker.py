import requests
import json
from deepdiff import DeepDiff

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49666794466612201538245840337915794979476686302022205442",
                "data": "eyJyaWRlIjogeyJQVUxvY2F0aW9uSUQiOiAxMzAsICJET0xvY2F0aW9uSUQiOiAyMDUsICJ0cmlwX2Rpc3RhbmNlIjogMy43NX0sICJyaWRlX2lkIjogMjU2fQ==",
                "approximateArrivalTimestamp": 1757056021.877
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49666794466612201538245840337915794979476686302022205442",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::123456789012:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:123456789012:stream/ride_events"
        }
    ]
}


url  = "http://localhost:8080/2015-03-31/functions/function/invocations"
actual_response = requests.post(url, json=event).json()
print("Actual response:")

print(json.dumps(actual_response, indent=2))

expected_response = {'predictions': 
                        [{
                            'model': 'ride-duration-prediction-model', 
                            'version': 'm-73b1fea3e7c0444ebff7192f9d16ed53', 
                            'prediction': {'ride_duration': 18.120189001540375, 'ride_id': 256}}
                         ]
                    }

diff = DeepDiff(actual_response, expected_response, significant_digits=3)
# print('diff=', diff)

# print(diff)

assert 'type_changes' not in diff
assert 'values_changed' not in diff


# assert  actual_response == expected_response