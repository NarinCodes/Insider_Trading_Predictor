"""
This module acquires the data and interacts with S3
"""
import typing
import time
import logging.config
import os
import re
from datetime import date
import requests
import boto3
import botocore
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

aws_id = os.environ.get('AWS_ACCESS_KEY_ID')  # AWS ID as environment variable
aws_key = os.environ.get('AWS_SECRET_ACCESS_KEY')  # AWS Key as environment variable

def get_transactions(endpoint:str,
                     save_path_1: str,
                     save_path_2: str,
                     representatives: typing.List[str],
                     tickers: typing.List[str],
                     attempts: int = 4,
                     wait: int = 3,
                     wait_multiple: int = 2) -> None:
    """
    Get the Congressional trade data from the House Stockwatcher API.
    Use exponential backoff while getting the sentiment data from the API.
    Convert the downloaded data to a DataFrame and save in specified path.
    Args:
        endpoint (str): URL to interface with the API
        save_path_1 (str): path where the API data will be saved as a DataFrame
        save_path_2 (str): path where the recent_transaction data will be saved as a DataFrame
        representatives (typing.List[str]): List of representatives to keep
        tickers (typing.List[str]): List of tickers to keep
        attempts (int): Maximum retry count
        wait (int): Delay period (start with 3 seconds)
        wait_multiple (int): Delay increase interval
    Returns:
        None
    """
    # Run the loop up until the specified limit is reached
    for i in range(attempts):
        try:
            logger.info('Obtaining data from the Stockwatcher API')
            url = endpoint
            response_init = requests.get(url)
            response = response_init.json()
            df = pd.DataFrame(response)
            df = df[df['representative'].isin(representatives)]
            recent_df = df[['representative','transaction_date',
                            'ticker','asset_description','amount','type']]
            df = df[df['ticker'].isin(tickers)]
            df.to_csv(save_path_1, index=False)
            recent_df.to_csv(save_path_2, index=False)
        # Try again if more attempts remain
        except requests.exceptions.ConnectionError as except_1:
            if i + 1 < attempts:
                logger.warning('There was a connection error during attempt %i of %i. '
                               'Waiting %i seconds then trying again.',
                               i + 1, attempts, wait)
                time.sleep(wait)
                # Keep increasing the wait times after each attempt
                wait = wait * wait_multiple
            else:
                logger.error(
                    'Connection error. The max number of attempts (%i) have been made to connect.'
                    'Please check your connection then try again',
                    attempts)
                raise except_1
        # Check for valid url as input
        except requests.exceptions.MissingSchema as except_2:
            logger.error('Need to add http:// to beginning of url. Url provided: %s', url)
            raise except_2
        else:
            logger.info('Stockwatcher data saved in %s', save_path_1)
            logger.info('Recent transaction data saved in %s', save_path_2)
            break

def get_stock_price(input_path:str,
                    output_path_1:str,
                    output_path_2:str,
                    tickers: typing.List[str]) -> None:
    """
    Obtain the stock price data from Yahoo finance API by looping over
    the DataFrame obtained from Stockwatcher API and obtaining the transaction
    dates

    Args:
        input_path (str): path to Stockwatcher API data
        output_path_1 (str): path to save historical stock price data
        output_path_2 (str): path to save current stock price data
        tickers (typing.List[str]): list of tickers in the Stockwatcher data
    Returns:
        None
    """
    df = pd.read_csv(input_path)
    purch_dates = df.groupby(['transaction_date', 'ticker']).size().reset_index(name='freq')
    purch_dates['trans_date'] = pd.to_datetime(purch_dates['transaction_date'])
    purch_dates['next_date'] = purch_dates['trans_date'] + pd.Timedelta(days=1)
    day_price = pd.DataFrame(columns=['ticker', 'date', 'price'])
    logger.info('Obtaining data from YFinance API. This may take a few minutes')

    for i in range(len(purch_dates)):
        ticker = yf.Ticker(purch_dates.loc[i,'ticker'])

        val = ticker.history(start = purch_dates.loc[i,'trans_date'],
                             end   = purch_dates.loc[i,'next_date'])
        row_1 = pd.DataFrame({'ticker':[purch_dates.loc[i,'ticker']],
                            'date':[purch_dates.loc[i,'transaction_date']],
                            'price':[val['Close'].values[0]]})
        day_price = pd.concat([day_price,row_1], axis = 0)

    current_price = pd.DataFrame(columns=['ticker', 'date','price'])
    for i in tickers:
        ticker = yf.Ticker(i)
        val = ticker.history(start=date.today().strftime('%Y-%m-%d'))
        row_2 = pd.DataFrame({'ticker':[i],
                            'date':[date.today().strftime('%Y-%m-%d')],
                            'price':[val['Close'].values[0]]})
        current_price = pd.concat([current_price,row_2], axis = 0)

    day_price = day_price.dropna()
    current_price = current_price.dropna()
    day_price.to_csv(output_path_1, index=False)
    current_price.to_csv(output_path_2, index = False)
    logger.info('YFinance historical stock-price data saved to %s', output_path_1)
    logger.info('YFinance current stock-price data saved to %s', output_path_2)

def parse_s3(s3path:str)->typing.Tuple[str,str]:
    '''
    Parses string to extract bucket name and s3 path
    Args:
        s3path (str): full s3 path
    Returns:
        s3bucket (str): name of s3 bucket
        s3path (str): directory path within s3 bucket
    '''
    regex = r's3://([\w._-]+)/([\w./_-]+)'
    matched = re.match(regex, s3path)  # matched groups based on regex string
    s3bucket = matched.group(1)
    s3path = matched.group(2)
    return s3bucket, s3path

def upload_s3(s3path: str, file_name:str, local_path:str) -> None:
    '''
    Uploads an input file to the specified S3 Bucket
    Args:
        local_path (str): the filepath location of file that will be uploaded
        file_name (str): the name of the input file being uploaded to S3
        s3path (str): the path to the user's AWS S3 bucket
    Returns:
        None
    '''
    session = boto3.Session(aws_access_key_id=aws_id,
                            aws_secret_access_key=aws_key)
    client = session.client('s3')
    if file_name == 'stockwatcher':
        s3path = s3path + '/data_new/stockwatcher.csv'
    elif file_name == 'transact_price':
        s3path = s3path + '/data_new/transact_price.csv'
    elif file_name == 'current_price':
        s3path = s3path + '/data_new/current_price.csv'
    elif file_name == 'recent_transactions':
        s3path = s3path + '/data_new/recent_transactions.csv'
    s3bucket, s3_just_path = parse_s3(s3path)

    try:
        client.upload_file(local_path, s3bucket, s3_just_path)
    except botocore.exceptions.NoCredentialsError:
        logger.error('Please provide AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY env vars.')
    except boto3.exceptions.S3UploadFailedError:
        logger.error('Please provide a valid S3 bucket name.')
    else:
        logger.info('Data successfully uploaded from %s to %s', local_path, s3path)


def download_s3(s3path:str,
                file_name:str,
                local_path:str,
                sep:str) -> None:
    '''Downloads file from S3
    Args:
        s3path (str): the path where the file will be located on s3
        file_name (str): the name of the file to be downloaded from s3
        local_path (str): the filepath location of file that will be downloaded to
        sep (str): separator for downloaded file
    Returns:
        None
    '''
    if file_name == 'stockwatcher':
        s3path = s3path + '/data_new/stockwatcher.csv'
    elif file_name == 'transact_price':
        s3path = s3path + '/data_new/transact_price.csv'
    elif file_name == 'current_price':
        s3path = s3path + '/data_new/current_price.csv'
    elif file_name == 'recent_transactions':
        s3path = s3path + '/data_new/recent_transactions.csv'

    try:
        df = pd.read_csv(s3path,sep=sep)
    except botocore.exceptions.NoCredentialsError:
        logger.error('Please provide AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY env vars.')
    else:
        df.to_csv(local_path, sep=sep, index=False)
        logger.info('Data downloaded from %s to %s', s3path, local_path)
