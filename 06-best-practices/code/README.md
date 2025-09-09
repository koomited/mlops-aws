```bash
    docker build -t stream-model-duration:v2 .
```

```bash
docker run -it --rm -p ride-duration-prediction-service:v2
```

```bash
docker run -it --rm \
  -p 8080:8080 \
  -e PREDICTIONS_STREAM_NAME="ride_predictions" \
  -e TEST_RUN="True" \
  stream-model-duration:v2
```

```bash
docker run -d -it --rm \
  -p 8080:8080 \
  -e PREDICTIONS_STREAM_NAME="ride_predictions" \
  -e MODEL_LOCATION="/app/model" \
  -e TEST_RUN="True" \
  -e RUN_ID="Test123" \
  -v $(pwd)/model:/app/model \
  stream-model-duration:v2
```