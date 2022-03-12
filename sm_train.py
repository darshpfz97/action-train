import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.analytics import TrainingJobAnalytics
from report import ResultReport
from datetime import datetime
from sagemaker.pytorch.estimator import PyTorch
import os

import pandas as pd
import sagemaker
import time

sagemaker_session = sagemaker.Session(boto3.session.Session())

# Put the right role and input data
# https://github.com/aws/sagemaker-python-sdk/issues/300
# -> https://github.com/aws/sagemaker-python-sdk/issues/300#issuecomment-409045648
role = "arn:aws:iam::137229062754:role/sagemaker1"
bucket = "gmaist345"
eval_path=f"s3://{bucket}/cresemba-kits/eval"
input_path = f"s3://{bucket}/Medical Insights_May2021_DFOs_SV_Cresemba_(Isavuconazole).xlsx"
cresemba_path=f"s3://{bucket}/utility_files/cresemba_KITS.txt"
output_path = f"s3://{bucket}/cresemba-kits/model"
code_path = f"s3://{bucket}/cresemba-kits/src"

# Make sure the metric_definition and its regex
# Train_epoch=1.0000;  Train_loss=0.8504;
# Test_loss=0.3227;  Test_accuracy=0.9100;
metric_definitions=[
                        {'Name': 'test:loss', 'Regex': 'Test_loss=(.*?);'},
                        {'Name': 'test:accuracy', 'Regex': 'Test_accuracy=(.*?);'},
                        {'Name': 'train:loss', 'Regex': 'Train_loss=(.*?);'},
                        {'Name': 'train:epoch', 'Regex': 'Train_epoch=(.*?);'}
                    ]

hyperparameters = {
    "input_path": input_path, # Where our model will read the training data.
    "cresemba_path": cresemba_path,
    "eval_path":eval_path
}
g1=time.time()*100000
g2=int(g1)
job_names=f"cresemba-{g2}"
estimator = PyTorch(
    entry_point='main_TransferLearning.py', # The name of our model script.
    source_dir='KIT_desc_classification/',
    instance_type='ml.p2.xlarge', # Instnace with GPUs.
    instance_count=1,
    framework_version='1.5.0', # PyTorch version.
    py_version='py3',
    hyperparameters=hyperparameters, # Passed as command-line args to entry_point.
    code_location=code_path, # Where our source_dir gets stored in S3.
    output_path=output_path, # Where our model outputs get stored in S3.

    role=role, # Role with SageMaker access.
    sagemaker_session=sagemaker_session
)

estimator.fit(inputs=None, job_name=job_names)


########################################################################
# DONOT EDIT AFTER THIS LINE
########################################################################
training_job_name = estimator.latest_training_job.name
    
# Get metric values
metric_names = [ metric['Name'] for metric in estimator.metric_definitions ] 
metrics_dataframe = TrainingJobAnalytics(training_job_name=training_job_name, metric_names=metric_names).dataframe()

# Report results
rr = ResultReport()
rr.report(estimator.model_data, metrics_dataframe)
    
# Update leaderboard. Make sure the key name is right
# Use any name if you don't want to use the leaderboard
score_metric = 'test:accuracy'
score_name = 'Test Accuracy'
leaderboard_ascending = False

if score_metric not in metric_names:
    print("leaderboard key name is not correct. No leaderboard support.")
    exit(-1)

accuracy_df = TrainingJobAnalytics(
    training_job_name=training_job_name, metric_names=[score_metric]).dataframe()

df_len = len(accuracy_df.index)
if df_len == 0:
    score = 0
else:  # Use the last value as the new score
    score = accuracy_df.loc[df_len-1]['value']

# Update new score to the leaderboard
rr.update_leaderboard(score, scoreText=score_name)
