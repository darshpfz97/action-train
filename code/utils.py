import joblib
import os
import json
import boto3
import pandas as pd
import io


def save_model(model, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    joblib.dump(model, file_path+file_name)

def save_predictions(df, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    df.to_csv(file_path+file_name)

def load_model(path):
    model = joblib.load(path)
    return model

def load_JSON(path):
    with open(path) as json_file:
        dict = json.load(json_file)
    return dict

def read_object(s3_path, file_type, **kwargs):
    """Read a file from S3 into a pandas.DataFrame.
    Arguments:
        s3_path (str)
    Returns:
        df (pd.DataFrame)
    """
    # Get S3 client.
    s3 = boto3.client('s3')
    # Get object.
    bucket, key = s3_path.split('/')[2], '/'.join(s3_path.split('/')[3:])
    object = s3.get_object(Bucket=bucket, Key=key)
    # Read object into pandas.DataFrame.
    if file_type == "csv":
        df = pd.read_csv(object['Body'], **kwargs)
    elif file_type == "json":
        df = pd.read_json(object['Body'], lines=True, **kwargs)
    elif file_type == "xlsx":
        file_content = object["Body"].read()
        read_excel_data = io.BytesIO(file_content)
        df = pd.read_excel(read_excel_data, engine="openpyxl", sheet_name="MEDIC_OneMed_DFO_Isavuconazole", skiprows=1,
                         keep_default_na=False)
    elif file_type=='txt':
        file_content1 = object["Body"]
        df = json.load(file_content1)
        
    else:
        raise Exception("Only file_type csv and json currently supported.")
    return df