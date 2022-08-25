"""
This is the calling script that leverages modules in the src folder
"""

import logging.config
import argparse
import yaml

from src.createdb    import (create_db,
                             add_df)
from src.clean       import (join_current_price,
                             join_transact_price,
                             add_response,
                             filter_df,
                             drop_dups,
                             impute_missing)
from src.train       import (train)
from src.acquire_new import (get_stock_price,
                             get_transactions,
                             upload_s3,
                             download_s3)

# use the config file for logging purposes
logging.config.fileConfig('config/logging/local.conf')

# define the argument parser
parser = argparse.ArgumentParser(description='Provide different arguments to run pipeline')

# add argument for config yaml file
parser.add_argument('--config', default='config/test.yaml',
                    help='Path to configuration file')

# allow for subparsers
subparsers = parser.add_subparsers(dest='subparser_name')

# subparser for creating a table
sb_create = subparsers.add_parser('create_table', description='Create a database table')

# subparser for ingesting data to the table
sb_ingest = subparsers.add_parser('ingest_data', description='Ingest data to the table')

# subparser for downloading API data and pushing to S3 bucket
sb_ingest_new = subparsers.add_parser('acquire_new', description='Add data to s3 bucket')
sb_ingest_new.add_argument('--s3_raw',
                           required=False,
                           help='Will load data to specified path',
                           default='')

# subparser for downloading raw data from S3 and creating cleaned data
sb_download = subparsers.add_parser('clean',
                                    description='Download & clean data from s3 bucket')               
sb_download.add_argument('--s3_raw',
                         required=False,
                         help='Will load data from specified path',
                         default='')

# subparser for creating features from the cleaned data
sb_add_features = subparsers.add_parser('add_features',
                                    description='Save the final DataFrame with all features')

# subparser for creating the model object and other artifacts
sb_get_model = subparsers.add_parser('get_model',
                                    description = 'Save all the modeling artifeacts')

# subparser for obtaining predictions
sb_get_preds = subparsers.add_parser('get_preds',
                                    description = 'Save all the predictions')

# subparser for generating metrics
sb_get_metrics = subparsers.add_parser('get_metrics',
                                    description = 'Save all the performance metrics')

# parse all the arguments
args = parser.parse_args()

# obtain the name of the subparser
sp_used = args.subparser_name

if __name__ == '__main__':
    with open(args.config, 'r', encoding='utf8') as f:
        y_conf = yaml.load(f, Loader=yaml.FullLoader)

    if sp_used == 'acquire_new':
        # get data from the APIs
        get_transactions(**y_conf['acquire_new']['get_transactions'])
        get_stock_price(**y_conf['acquire_new']['get_stock_price'])

        # push the raw data to S3
        upload_s3(args.s3_raw,**y_conf['acquire_new']['upload_s3']['recent_transactions'])
        upload_s3(args.s3_raw,**y_conf['acquire_new']['upload_s3']['stockwatcher'])
        upload_s3(args.s3_raw,**y_conf['acquire_new']['upload_s3']['current_price'])
        upload_s3(args.s3_raw,**y_conf['acquire_new']['upload_s3']['transact_price'])

    elif sp_used == 'create_table':
        create_db()

    elif sp_used == 'ingest_data':
        add_df(y_conf['create_db']['local_path'])

    elif sp_used == 'clean':
        # download data from S3
        download_s3(args.s3_raw,**y_conf['clean']['download_s3']['rt'])
        download_s3(args.s3_raw,**y_conf['clean']['download_s3']['sw'])
        download_s3(args.s3_raw,**y_conf['clean']['download_s3']['cp'])
        download_s3(args.s3_raw,**y_conf['clean']['download_s3']['tp'])

        # create the cleaned data
        data = join_transact_price(**y_conf['clean']['transact'])
        join_current_price(data, **y_conf['clean']['current'])

    elif sp_used == 'add_features':
        # create the features needed for modeling
        data = add_response(**y_conf['clean']['add_response'])
        data = filter_df(data, **y_conf['clean']['filter'])
        data = drop_dups(data)
        impute_missing(data, **y_conf['clean']['impute_missing'])

    elif sp_used == 'get_model':
        # save the model and other required artifacts
        train(**y_conf['train']['get_model'])

    elif sp_used == 'get_preds':
        # obtain the predictions
        train(**y_conf['train']['get_preds'])

    elif sp_used == 'get_metrics':
        # obtain the performance metrics
        train(**y_conf['train']['get_metrics'])

    else:
        parser.print_help()
