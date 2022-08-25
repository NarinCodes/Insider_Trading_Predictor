python3 run.py acquire_new --s3_raw $S3_BUCKET
python3 run.py clean --s3_raw $S3_BUCKET
python3 run.py add_features
python3 run.py get_model
python3 run.py get_preds
python3 run.py get_metrics