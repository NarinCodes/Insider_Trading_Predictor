# Table of Contents
* [Project Charter](#Project-Charter)
* [Directory structure ](#Directory-structure)
* [Instructions to execute the Pipeline](#Instructions-to-execute-the-Pipeline)
* [Instructions to connect to a Database](#Connecting-to-a-Database)
* [Running the app ](#Running-the-app)
	* [1. Build the Image ](#1.-Build-the-Image)
	* [2. Run the Container ](#2.-Run-the-Container)
	* [3. Use the Flask app ](#3.-Use-the-Flask-app)
* [Testing](#Testing)

## Project Charter

## Identify Possible Instance of Insider Trading by Members of U.S. Congress

Developer: Narin Dhatwalia

![enter image description here](https://images.mktw.net/im-474636?width=700&size=1.4382022471910112&pixel_ratio=2)

### Background

Congress resembled a Wall Street trading desk last year, with lawmakers making an estimated total of $355 million worth of stock trades, buying and selling shares of companies based in the U.S. and around the world.

At least 113 lawmakers have disclosed stock transactions that were made in 2021 by themselves or family members, according to a Capitol Trades analysis of disclosures and MarketWatch reporting. U.S. lawmakers bought an estimated $180 million worth of stock last year and sold $175 million.

The trading action taking place in both the House and the Senate comes as some lawmakers push for a ban on congressional buying and selling of individual stocks. Stock trading is a bipartisan activity in Washington, widely conducted by both Democrats and Republicans, the disclosures show. Congress as a whole tended to be slightly bullish last year with more buys than sells as the S&P 500 SPX soared and returned 28.4%. Republicans traded a larger dollar amount overall — an estimated $201 million vs. Democrats’ $154 million.

Stock picking by elected officials gets worrisome because there is widespread concern that legislators may have access to insider information. It is also possible that their stock purchases will consciously or unconsciously impact policy making.

### Vision 

The aim of the project is to predict the probability of short-term increase in a company's stock price after a U.S. Congress official has invested. The final goal is to build a model which can accurately determine whether an investment made by a U.S. lawmaker will lead to immediate increase in the company's stock price. Results from this model can benefit retail investors while picking their own stocks, as well as watchdog groups to explore possible cases of insider trading.

### Mission

A logistic regression model is leveraged for the purpose of binary classification. A response value of 1 is assigned if the company's stock is trading at a higher price today than on the day of the transaction (i.e. when the Congress official made the purchase/sale).

Two APIs are used to source the data for this project. The "House Stock Watcher" API is used to extract stock transactions of Congress members (from documents filed under the Stock Act of 2012), and the "Yahoo Finance" API is used to obtain the prices at which different stocks were trading on a specific day.

Once the app is live, users can obtain a probability of the transaction resulting in short-term increase of stock price. This information can be vital for retail investors who track financial disclosures made by Congress members, and wish to assess whether they should also purchase the same stock. Moreover, watchdog groups can possibly interpret a very high probability as a possible suggestion of insider trading.

Features to be input by users once the model is live:

* Name of Congress Official
* Type of transaction (e.g. purchase, sale, etc.)
* Ownership of the stock (e.g. self, dependent, etc.)
* Ticker of the company they've invested in
* Dollar amount that has been invested
* Current price of the company's stock

House Stock Watcher: https://housestockwatcher.com/

Yahoo Finance: https://pypi.org/project/yfinance/

### Success Criteria

The two success criteria for this project are as follows:

* The prediction accuracy and the AUC/ROC score of the binary classifier are concrete indicators of the model's predictive performance. Before the model goes live, the prediction accuracy should be above 0.70 and the AUC/ROC score should be above 0.85.

* Once the model goes live, it becomes important to measure user engagement. Therefore, the business outcomes of concern will be the number of predictions made per day, number of website visits per day, and the percentage of repeat visitors (i.e. users that return to the webpage within the same week).


## Directory structure 

```
├── README.md                         <- You are here
├── app
│   ├── static/                       <- CSS, JS files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs   
│
├── config                            <- Directory for configuration files 
│   ├── local/                        <- Directory for env variables & other local configurations that *don't sync** to Github 
│   ├── logging/                      <- Configuration of python loggers
│   ├── flaskconfig.py                <- Configurations for Flask API
│   ├── test.yaml		      <- YAML file for passing arguments arguments to the calling script
│
├── data                              <- Folder that contains data used or generated
│   ├── external/                     <- Data downloaded directly from the APIs (files saved here in step 1 of the pipeline)
│   ├── clean/                        <- Intermediate datasets created while running the pipeline (most files get saved here)
│   ├── s3_downloads/		      <- Data downloaded directly from AWS S3 (these files get used in the pipeline)
│
├── deliverables/                     <- Presentation slides saved here in both PDF and PPT file format
│
├── dockerfiles/                      <- Directory for all project-related Dockerfiles 
│   ├── Dockerfile.app                <- Dockerfile for building image to run web app
│   ├── Dockerfile                    <- Dockerfile for building image to execute run.py  
│   ├── Dockerfile.test               <- Dockerfile for building image to run unit tests
│   ├── Dockerfile.pipeline	      <- Dockerfile for building image to run complete pipeline
│
├── figures/                          <- ROCAUC curve and Confusion Matrix get saved here
│
├── models/                           <- Trained model objects (TMOs), model predictions, and/or model summaries
│
├── notebooks/
│   ├── archive/                      <- Develop notebooks no longer being used.
│   ├── deliver/                      <- Notebooks shared with others / in final state
│   ├── develop/                      <- Current notebooks being used in development.
│
├── src/                              <- Source data for the project. No executable Python files should live in this folder.  
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── app.py                            <- Flask wrapper for running the web app 
├── run.py                            <- Simplifies the execution of one or more of the src scripts  
├── requirements.txt                  <- Python package dependencies 
├── .pylintrc                         <- Configuration file to customize Pylint functionality
├── run-pipeline.sh                   <- Shell script to run the entire modeling pipeline
├── .gitignore                        <- Used for specifying which files not to track via the Git workflow
```
## Instructions to execute the Pipeline

### Provide Environment Variables

Before running any of the steps described below, please provide the following environment variables:

1. AWS_ACCESS_KEY_ID
2. AWS_SECRET_ACCESS_KEY
3. SQLALCHEMY_DATABASE_URI
4. S3_BUCKET

The way to do so, is by entering the following command in terminal:

```
export S3_BUCKET='s3://bucket-name'
```

This will have to be done for each of the 4 variables listed above.

### For Sequentially Executing the Steps:

Please follow the steps below in their listed order. Run these steps in a terminal window from the root directory of the project repo.

### 1. Build the Docker Image

```
docker build -f dockerfiles/Dockerfile -t final-project .
```

### 2. Download the raw data and upload to S3

```
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET --mount type=bind,source="$(pwd)",target=/app/ final-project acquire_new --s3_raw $S3_BUCKET
```

### 3. Download the data from S3 and clean

```
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET --mount type=bind,source="$(pwd)",target=/app/ final-project clean --s3_raw $S3_BUCKET
```

### 4. Create the features

```
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project add_features
```

### 5. Get the model object and other artifacts

```
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project get_model
```

### 6. Get the predicted classes and probabilities

```
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project get_preds
```

### 7. Get the performance metrics

```
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project get_metrics
```

### For Running the entire Pipeline:

Please follow the 2 steps provided below

### 1. Build the Pipeline Docker Image

```
docker build -f dockerfiles/Dockerfile.pipeline -t final-project-pipeline .
```

### 2. Run the Pipeline Container

```
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET --mount type=bind,source="$(pwd)",target=/app/ final-project-pipeline run-pipeline.sh
```

## Connecting to a Database

If the __SQLALCHEMY_DATABASE_URI__ is provided as an environment variable, the connected database will either be an AWS RDS instance or a locally created SQLite database (depending on the environment variable). If no environment variable is provided, a local SQLite database will be used by default. 

The next two steps assume that the Docker Image for sequential execution has already been built. If that is not the case, please execute Step 1 in the Sequential Execution steps described above before continuing.

### 1. Run the Docker Container to Create Table

```
docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)",target=/app/ final-project create_table
```

### 2. Run the Docker Container to Ingest Data

```
docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)",target=/app/ final-project ingest_data
```

## Running the app 

### 1. Build the Image

To build the image, run from this directory (the root of the repo): 

```
docker build -f dockerfiles/Dockerfile.app -t final-project-app .
```

### 2. Run the Container

```
docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)",target=/app/ -p 5000:5000 final-project-app
```

### 3. Use the Flask app

Open your browser and type __http://localhost:5000/__ in the address bar. You should be able to interact with the app at this point. Try entering different inputs and obtaining a prediction by clicking on the button at the bottom of the page.

## Testing

Create the Docker Image for Unit Tests:

```
docker build -f dockerfiles/Dockerfile.test -t final-project-tests .
```

To run the tests, enter the following command: 

```
docker run final-project-tests
```
