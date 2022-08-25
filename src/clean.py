"""
This module processes and cleans the raw data to prepare it for modeling
"""
import logging
import typing
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# function to join the transaction data with historical stock price data
def join_transact_price(input_path_1:str,
                        input_path_2:str) -> pd.core.frame.DataFrame:
    """
    This function joins the transaction data with the transaction-day stock
    price data by inner-joining the two DataFrames on the date and ticker columns

    Args:
        input_path_1 (str): path to the transaction data
        input_path_2 (str): path to the stock price data
    Returns:
        new_df (pd.core.frame.DataFrame): new DataFrame with transactions and stock price
        on the day of the transaction
    """
    # read-in the transaction data and stock price data from the specified paths
    try:
        data = pd.read_csv(input_path_1)
        data_2 = pd.read_csv(input_path_2)
    except FileNotFoundError:
        logger.error('File not found')
        logger.debug('Check path in the configuration file')

    transact_price = data_2.drop_duplicates()
    try:
        transact_price = transact_price[['ticker','date','price']]
    except KeyError as error3:
        logger.error('The stock-price data does not contain the required columns: %s', error3)
        raise error3
    # add transaction day stock-price
    new_df = pd.merge(data,
                      transact_price[~np.isnan(transact_price['price'])],
                      how='inner',
                      left_on=['ticker','transaction_date'],
                      right_on = ['ticker','date'])

    new_df = new_df.rename(columns={'price': 'trans_price'})
    logger.info('Join #1 of the DataFrames completed successfully')
    return new_df

# function to join the transaction data with current stock price data
def join_current_price(data: pd.core.frame.DataFrame,
                       local_path: str,
                       output_path: str) -> None:
    """
    This function joins the transaction data with the current-day stock
    price data by inner-joining the two DataFrames on the ticker column

    Args:
        data (pd.core.frame.DataFrame): DataFrame containing transactions and historical prices
        local_path (str): path to the current-day stock price data
        output_path (str): path to save the output DataFrame
    Returns:
        None
    """
    # read-in the stock price data from the specified path
    try:
        data_2 = pd.read_csv(local_path)
    except FileNotFoundError:
        logger.error('File not found')
        logger.debug('Check path in the configuration file')

    current_price = data_2.drop_duplicates()

    try:
        current_price = current_price[['ticker','date','price']]
    except KeyError as error4:
        logger.error('The stock-price data does not contain the required columns: %s', error4)
        raise error4
    # add current stock-price
    new_df = pd.merge(data,
                      current_price[~np.isnan(current_price['price'])],
                      how='inner',
                      left_on=['ticker'],
                      right_on = ['ticker'])

    new_df = new_df.rename(columns={'price': 'current_price'})
    logger.info('Join #2 of the DataFrames completed successfully')

    if output_path:
        new_df.to_csv(output_path, index=False)
        logger.info('Cleaned DataFrame saved to: %s', output_path)

# function to create the response variable
def add_response(input_data:typing.Union[str,pd.core.frame.DataFrame]) -> pd.core.frame.DataFrame:
    """
    This function creates the response variable by comparing historical
    stock price and current stock price. It assigns a value of 1 is the
    current price is higher than the historical price, else it assigns 0

    Args:
        input_data (typing.Union[str,pd.core.frame.DataFrame]): path where the input
        DataFrame should be obtained from, or alternatively, an actual DataFrame object
    Returns:
        data (pd.core.frame.DataFrame): DataFrame with the response included
    """
    if isinstance(input_data, str):
        try:
            data = pd.read_csv(input_data)
        except FileNotFoundError:
            logger.error('File not found')
            logger.debug('Check path in the configuration file')
    elif isinstance(input_data, pd.core.frame.DataFrame):
        data = input_data

    try:
        # add binary variable for response
        data['response'] = data['current_price'] > data['trans_price']
    except KeyError as error1:
        logger.error('The response can not be created due to missing columns: %s', error1)
        raise error1
    # change the response variable type from boolean to int
    data['response'] = data['response']*1
    logger.info('Response variable added to the DataFrame successfully')
    return data

# function to exclude certain variables
def filter_df(data: pd.core.frame.DataFrame,
              columns: typing.List[str]) -> pd.core.frame.DataFrame:
    '''
    This function filters the DataFrame by excluding certain columns. The
    columns to be excluded are provided as user-input within a list of strings
    where each string in the list is the name of a column

    Args:
        data (pd.core.frame.DataFrame): input DataFrame with all columns present
        columns (typing.List[str]): the list of column names to be excluded
    Returns:
        new_df (pd.core.DataFrame): output DataFrame with certain columns filtered out
    '''
    try:
        new_df = data.drop(columns=columns)
        logger.info('DataFrame filtered successfully')
    except KeyError:
        logger.warning('Some columns are not in the DataFrame. The unfiltered DataFrame will be used')
        new_df = data
    return new_df

# function to drop duplicates from the DataFrame
def drop_dups(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    This function drops rows which are entirely duplicates (note that
    all columns are used while determining whether a row is a duplicate)

    Args:
        data (pd.core.frame.DataFrame): input DataFrame that may have duplicates
    Returns:
        new_df (pd.core.frame.DataFrame): output DataFrame with all duplicates removed
    """
    try:
        new_df = data.drop_duplicates()
        records_droped = data.shape[0] - new_df.shape[0]
        logger.info('Duplicates dropped successfully. Num of duplicates = %s', records_droped)
    except AttributeError as error2:
        logger.error('Must pass a DataFrame to the `data` argument: %s', error2)
        raise error2
    return new_df

# function to impute missing values in the DataFrame
def impute_missing(data:pd.core.frame.DataFrame,
                   save_path:str = None,
                   column:str = 'owner',
                   replacement:str = 'undisclosed',
                   missing_val:str = '--') -> pd.core.frame.DataFrame:
    """
    This function identifies all rows with null values or other placeholders
    that indicate missing data. Next, the identified rows are imputed with
    a replacement value before saving the cleaned data to the user
    specified path

    Args:
        data (pd.core.fram.DataFrame): DataFrame with missing values
        save_path (str): path to save the cleaned DataFrame
        column (str): column name of feature containing missing values
        replacement (str): value to be used in place of missing data
        missing_val (str): placeholder that indicates missing data

    Returns:
        data (pd.core.frame.DataFrame) : output DataFrame with imputed values
    """
    try:
        data[column] = data[column].fillna(replacement)
        data[column] = data[column].replace({missing_val:replacement})
        logger.info('Missing values imputed successfully')
    except KeyError:
        logger.warning('The column to be imputed does not exist. Using the original DataFrame')

    if save_path:
        data.to_csv(save_path, index = False)
        logger.info('DataFrame with features saved to: %s', save_path)
    return data
