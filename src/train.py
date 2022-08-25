"""
This module contains functions to:
1. One-hot encode & standard scale the data, train model for binary classification, save outputs
2. Helper function to split the data, Train the model, save the modeling outputs
3. Get the model, scaler, and encoder from specified paths
4. Transform user-input into an input accepted by the model
5. Make prediction on a single row of user input after transforming
"""
import logging
import warnings
import pickle
import typing
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import sklearn.preprocessing as skp

logger = logging.getLogger(__name__)

def train(local_path: str,
          categ: typing.List[str],
          response: str,
          results_path: str,
          matrix_path: str,
          roc_path: str,
          model_path: str,
          encoder_path: str,
          scaler_path: str,
          test_size: float,
          random_state: int,
          max_iter: int,
          output_data_path:str,
          pred_path_1:str,
          pred_path_2:str) -> None:
    '''
    This function One-Hot encodes & Standard Scales the data. Next, train-test split
    and model training steps are executed. Finally, all the modeling outputs get written to
    the specified paths. Internally, this function calls the train_evaluate function.

    Args:
        local_path (str): path to cleaned data
        categ (typing.List[str]): list of column names representing categorical features
        response (str): column name of response variable
        results_path (str): path to write yaml file with model evaluation results
        matrix_path (str): path to write png image with test set confusion matrix
        roc_path (str): path to write png image with test set AUCROC curve
        model_path (str): path to write pickle object wth Logistic Regression model
        encoder_path (str): path to write pickle object wth fitted One-Hot encoder
        scaler_path (str): path to write pickle object wth fitted Standard Scaler
        test_size (float): fraction of original data to split into test set
        random_state (int): random state for training model
        max_iter (int): maximum number of iterations taken for the solvers to converge
        output_data_path (str): path to save x_train, x_test, y_train, y_test
        pred_path_1 (str): path to save predicted classes
        pred_path_2 (str): path to save predicted probabilities

    Returns:
        None
    '''
    try:
        data = pd.read_csv(local_path)
    except FileNotFoundError:
        logger.error("File %s not found at ", local_path)
        logger.debug("Check path in the configuration file")
    enc = OneHotEncoder().fit(data[categ])
    dummy_categ = enc.transform(data[categ])
    dummy_categ = pd.DataFrame(dummy_categ.toarray())
    features = pd.concat([dummy_categ, data.drop(categ+[response], axis=1)], axis=1)
    features.columns = enc.get_feature_names_out().tolist() +\
                         data.drop(categ+[response], axis=1).columns.to_list()
    response = data[response].values.ravel()

    model, scaler, x_train, x_test, y_train, y_test = train_evaluate(features,
                                   response,
                                   results_path,
                                   matrix_path,
                                   roc_path,
                                   test_size,
                                   random_state,
                                   max_iter,
                                   pred_path_1,
                                   pred_path_2)

    if output_data_path:
        pd.DataFrame(x_train).to_csv(output_data_path+"/x_train.csv", index=False)
        pd.DataFrame(x_test).to_csv(output_data_path+"/x_test.csv", index=False)
        pd.DataFrame(y_train).to_csv(output_data_path+"/y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv(output_data_path+"/y_test.csv", index=False)
        logger.info("Data after train/test split saved in %s folder", output_data_path)

    if model_path and encoder_path and scaler_path:
        pickle.dump(model, open(model_path, "wb"))
        logger.info("Model saved to: %s", model_path)
        pickle.dump(enc, open(encoder_path, "wb"))
        logger.info("OneHotEncoder saved to: %s", encoder_path)
        pickle.dump(scaler, open(scaler_path, "wb"))
        logger.info("StandardScaler saved to: %s", scaler_path)

def train_evaluate(features: pd.core.frame.DataFrame,
                   response: np.ndarray,
                   results_path: str,
                   matrix_path: str,
                   roc_path: str,
                   test_size: float,
                   random_state: int,
                   max_iter: int,
                   pred_path_1: str,
                   pred_path_2: str) -> typing.Tuple[typing.Union[sk._logistic.LogisticRegression,
                                                   skp._data.StandardScaler,
                                                   np.ndarray]]:
    '''
    This function train-test splits the data, builds the model, evaluates the
    model performance, then outputs the ROCAUC Curve png, Confusion Matrix png,
    and Evaluation Metrics yaml to the specified paths

    Args:
        features (pd.core.frame.DataFrame): DataFrame holding feature variables
        response (numpy.ndarray): array holding responses for each individual
        results_path (str): path to write yaml file with model evaluation results
        matrix_path (str): path to write png image with confusion matrix
        roc_path (str): path to write png image with AUCROC curve
        test_size (float): fraction of original data to split into test set
        random_state (int): random state for training model
        max_iter (int): maximum number of iterations taken for the solvers to converge
        pred_path_1 (str): path for saving predicted classes
        pred_path_2 (str): path for saving predicted probabilities

    Returns:
        log_reg (sk._logistic.LogisticRegression): binary logistic regression classifier object
        scaler (skp._data.StandardScaler): fitted standard scaler object
        x_train (np.ndarray): independent variables in the training data
        x_test (np.ndarray): independent variables in the test data
        y_train (np.ndarray): dependent variable in the training data
        y_test (np.ndarray): dependent variable in the test data
    '''
    x_train, x_test, y_train, y_test = train_test_split(features, response,
                                                        test_size=test_size,
                                                        random_state=random_state)
    model = LogisticRegression(max_iter=max_iter,random_state=random_state)
    logger.debug("Model training")
    scaler = StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test  = scaler.transform(x_test)

    x_train = pd.DataFrame(scaled_x_train, index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaled_x_test, index=x_test.index, columns=x_test.columns)

    log_reg = model.fit(x_train, y_train)
    ypred_bin_test = log_reg.predict(x_test)
    ypred_proba_test = log_reg.predict_proba(x_test)

    if pred_path_1 and pred_path_2:
        pd.DataFrame(ypred_bin_test).to_csv(pred_path_1,index=False)
        pd.DataFrame(ypred_proba_test).to_csv(pred_path_2,index=False)
        logger.info("Saved test-set predicted classes to %s", pred_path_1)
        logger.info("Saved test-set predicted probabilities to %s", pred_path_2)

    auc = roc_auc_score(y_test, ypred_proba_test[:, 1])
    loss = log_loss(y_test, ypred_proba_test)
    creport = classification_report(y_test, ypred_bin_test,output_dict=True)

    if matrix_path:
        ConfusionMatrixDisplay.from_estimator(log_reg, x_test, y_test)
        plt.savefig(matrix_path)
        logger.info("Confusion matrix saved to: %s", matrix_path)
    if roc_path:
        RocCurveDisplay.from_estimator(log_reg, x_test, y_test)
        plt.savefig(roc_path)
        logger.info("AUCROC curve saved to: %s", roc_path)

    flat_list = [item for items in model.coef_.tolist() for item in items]
    coeffs = dict(zip(x_train.columns.tolist(), flat_list))
    results = [creport,{"AUC": str(auc), "Log Loss": str(loss),"Coefficients" : coeffs}]

    if results_path:
        with open(results_path, "w",encoding="utf8") as file:
            yaml.dump(results, file)
        logger.info("Model results written to: %s", results_path)

    return log_reg, scaler, x_train, x_test, y_train, y_test

def get_model(model_path:str,
              encoder_path:str,
              scaler_path:str) -> typing.Tuple[sk._logistic.LogisticRegression,
                                               skp._encoders.OneHotEncoder,
                                               skp._data.StandardScaler]:
    '''
    Fetches the pickled model, encoder, scalar from the specified paths

    Args:
        model_path (str): path to pickled model
        encoder_path (str): path to pickled encoder
        scaler_path (str): path to pickled standard scaler
    Returns:
        model (sk._logistic.LogisticRegression): binary classifier logistic regression model
        encoder (skp._encoders.OneHotEncoder): encoder for categorical variables
        scaler (skp._data.StandardScaler): standard scaler for preprocessing
    '''
    try:
        with open(model_path, "rb") as input_file:
            model = pickle.load(input_file)

        with open(encoder_path, "rb") as input_file:
            enc = pickle.load(input_file)

        with open(scaler_path, "rb") as input_file:
            scaler = pickle.load(input_file)

    except FileNotFoundError:
        logger.error("File %s not found at ", model_path)
        logger.debug("Check path in the configuration file")

    return model, enc, scaler

def transform(encoder:skp._encoders.OneHotEncoder,
              scaler:skp._data.StandardScaler,
              cat_inputs:typing.List[str],
              trans_price:float) -> typing.List[typing.Union[int,float]]:
    '''
    Transforms raw input into one-hot encoded and standard scaled input for making
    predictions using the Logistic Regression model

    Args:
        encoder (skp._encoders.OneHotEncoder): encoder for categorical variables
        scaler (skp._data.StandardScaler): standard scaler for preprocessing
        cat_inputs (typing.List[str]): categorical inputs of transaction
        trans_price (float): stock price on the day of transaction
    Returns:
        test_new (typing.List[Union[int,float]]): encoded inputs for model prediction
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_new = encoder.transform([cat_inputs]).toarray()  # needs 2d array
        test_new = np.append(test_new[0], trans_price)  # encoder returns 2d array, need element inside
        test_new = [test_new]  # predict function expects 2d arrray
        test_new = scaler.transform(test_new)
    return test_new

def predict_ind(model:sk._logistic.LogisticRegression,
                encoder:skp._encoders.OneHotEncoder,
                scaler:skp._data.StandardScaler,
                cat_inputs: typing.List[str],
                trans_price:float) -> np.ndarray:
    '''
    Predicts the probabilities for a new row of input data provided by the user

    Args:
        model (sk._logistic.LogisticRegression): binary logistic regression model
        encoder (skp._encoders.OneHotEncoder): one-hot encoder for categorical variables
        scaler (skp._data.StandardScaler): standard scaler for preprocessing
        cat_inputs (typing.List[str]): categorical inputs of stock transaction
        trans_price (float): price of stock on day of trade
    Returns:
        prediction (numpy.ndarray): probability of short term increase in stock price
    '''
    test_new = transform(encoder, scaler, cat_inputs, trans_price)
    prediction = model.predict_proba(test_new)
    prediction = round(float(prediction[0][1]), 3)
    return prediction
