import os

import model

RUN_ID = os.getenv("RUN_ID", "m-73b1fea3e7c0444ebff7192f9d16ed53")

TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME", "ride_predictions")


model_service = model.init(
    prediction_stream_name=PREDICTIONS_STREAM_NAME,
    run_id=RUN_ID,
    test_run=TEST_RUN,
)


def lambda_handler(event, context):
    # pylint:disable=unused-argument
    return model_service.lambda_handler(event)
