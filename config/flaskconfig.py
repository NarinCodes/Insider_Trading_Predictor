"""
Config file for Flask app
"""
import os
DEBUG = True
LOGGING_CONFIG = 'config/logging/local.conf'
PORT = 5000
APP_NAME = 'Stockwatcher'
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = '0.0.0.0'
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100
MODEL_PATH = './models/model.pkl'
ENCODER_PATH = './models/encoder.pkl'
SCALER_PATH = './models/scaler.pkl'

# Connection string
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
if SQLALCHEMY_DATABASE_URI is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/Stockwatcher.db'
else:
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
