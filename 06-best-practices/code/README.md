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