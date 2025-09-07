
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pickle
import mlflow
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import argparse




from pathlib import Path
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

def read_data(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)
    df = df[((df.duration >= 1) & (df.duration <= 60))]
    categorical = ["PULocationID", "DOLocationID"]
    # numerical = ["trip_distance"]
    # df[categorical] = df[categorical].astype(str)
    df.loc[:, categorical] = df[categorical].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    
    return df

df_train = read_data(year=2021, month=1)
df_val = read_data(year=2021, month=2)
 


def create_X(df, dv=None):
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    data_dicts = df[categorical+numerical].to_dict(orient = "records")
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X= dv.fit_transform(data_dicts)
    else:
        X = dv.transform(data_dicts)
    
    return X, dv



def train_model(X_train, y_train, X_val, y_val, dv):

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        params = {"learning_rate":0.33707810007380146,
        "max_depth": 39,
        "min_child_weight": 1.3396380858101118,
        "objective": "reg:linear",
        "reg_alpha":0.21705039846156954,
        "reg_lambda": 0.06884936106500567,
        "seed":42
        }
        mlflow.log_params(params)

        booster = xgb.train(
                    params=params,
                    dtrain = train,
                    num_boost_round=300,
                    evals=[(valid, "validation")],
                    early_stopping_rounds=50
                )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
    # return run id from mlflow
        run_id = run.info.run_id
        print(f"Model saved in run {run_id}")
    return run_id
            

def run(year, month):
    df_train = read_data(year=year, month=month)
    
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    

    df_val = read_data(year=next_year, month=next_month)
    
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    return run_id


if __name__ == "__main__":
    # use argparse to get year and month
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--month", type=int, default=1)
    args = parser.parse_args()
    
    year = args.year
    month = args.month
    run_id = run(year, month)
    
    with open("models/latest_run_id.txt", "w") as f_out:
        f_out.write(run_id)
