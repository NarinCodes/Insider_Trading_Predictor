"""
This module defines the unit tests for clean.py
"""
import pandas as pd
import numpy as np
import pytest
from src import clean

# create sample data to check the functions
df_orig = pd.DataFrame({'current_price':[10,20,30,10],
                          'trans_price':[10,25,15,10],
                               'ticker':['AAPL','GOOG','MSFT','AAPL'],
                               'owner':['self','--',np.nan,'self']})

# define unit tests with happy paths
def test_add_response():
    """
    Check whether the actual response column is the same as expected
    """
    temp_df = df_orig.copy()
    actual_df = clean.add_response(temp_df)
    actual_response = actual_df['response']
    expected_response = pd.Series([0,0,1,0])
    assert actual_response.to_list() == expected_response.to_list()

def test_filter_df():
    """
    Check whether the filtered DataFrame is the same as expected
    """
    new_df = clean.filter_df(df_orig.copy(), ['trans_price', 'ticker'])
    expected_df = pd.DataFrame({'current_price':[10,20,30,10],
                                'owner':['self','--',np.nan,'self']})
    assert new_df.equals(expected_df)

def test_drop_dups():
    """
    Check whether the DataFrame without duplicates is the same as expected
    """
    output_df = clean.drop_dups(df_orig.copy())
    expected_df_2 = pd.DataFrame({'current_price':[10,20,30],
                          'trans_price':[10,25,15],
                               'ticker':['AAPL','GOOG','MSFT'],
                               'owner':['self','--',np.nan]})
    assert output_df.equals(expected_df_2)

def test_impute_missing():
    """
    Check whether the imputed DataFrame is the same as expected
    """
    imputed_df = clean.impute_missing(df_orig.copy())
    expected_df_3 = pd.DataFrame({'current_price':[10,20,30,10],
                          'trans_price':[10,25,15,10],
                               'ticker':['AAPL','GOOG','MSFT','AAPL'],
                               'owner':['self','undisclosed','undisclosed','self']})
    assert imputed_df.equals(expected_df_3)

# define unhappy paths
def test_add_response_unexpected_array():
    """
    Provide numpy array instead of DataFrame to add_response function
    """
    with pytest.raises(UnboundLocalError):
        clean.add_response(np.array(df_orig.copy()))

def test_filter_df_unexpected_array():
    """
    Provide numpy array instead of DataFrame to filter_df function
    """
    with pytest.raises(AttributeError):
        clean.filter_df(np.array(df_orig.copy()), ['trans_price', 'ticker'])

def test_filter_df_unexpected_column():
    """
    Provide `None` type input column names to the filter_df function
    """
    with pytest.raises(ValueError):
        clean.filter_df(df_orig.copy(), columns=None)

def test_drop_dups_unexpected_array():
    """
    Provide numpy array instead of DataFrame to the drop_dups function
    """
    with pytest.raises(AttributeError):
        clean.drop_dups(np.array(df_orig.copy()))

def test_impute_missing_unexpected_array():
    """
    Provide numpy array instead of DataFrame to the impute_missing function
    """
    with pytest.raises(IndexError):
        clean.impute_missing(np.array(df_orig.copy()))

def test_impute_missing_unexpected_column():
    """
    Provide unexpected input to replacement arg of the impute_missing function
    """
    with pytest.raises(TypeError):
        clean.impute_missing(df_orig.copy(),replacement=[3,4,5])
