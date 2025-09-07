#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
from sklearn.pipeline import make_pipeline
import os
import uuid
import sys

def generate_uuid(n):
    rides_ids = []
    for _ in range(n):
        rides_ids.append(str(uuid.uuid4()))
    return rides_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['ride_id'] = generate_uuid(len(df))
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id: str):
    logged_model = f's3://koomi-mlflow-artifacts-remote/2/models/{run_id}/artifacts'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def apply_model(input_file: str, run_id: str, output_file: str):
        
    print("Reading data...")
    df = read_dataframe(input_file)
    print("Preparing features...")
    dicts = prepare_dictionaries(df)
    print("Loading model...")
    model = load_model(run_id)
    print("Predicting...")
    y_pred = model.predict(dicts)
    print("Saving results...")
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    df_result.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")
    print("Done.")



def run():
    taxi_type =  sys.argv[1] 
    year = int(sys.argv[2]) 
    month = int(sys.argv[3])
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02}.parquet'
    RUN_ID = os.getenv('RUN_ID', 'm-73b1fea3e7c0444ebff7192f9d16ed53')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    apply_model(input_file, 
                RUN_ID, 
                output_file
            )

if __name__ == '__main__':
    run()




