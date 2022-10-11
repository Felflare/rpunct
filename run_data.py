import pandas as pd
import argparse
from training.prep_data import e2e_data


# parser
parser = argparse.ArgumentParser(description='Curate training, validation, and testing data to be fed into the model.')

# arguments
parser.add_argument(
    '-d',
    '--data',
    metavar='DATA',
    type=str,
    choices=['news', 'reviews'],
    default='news',
    help="Specify the dataset to be used to test the model: BBC News or Yelp reviews"
)

# parse arguments
args = parser.parse_args()
print(f"RUNNING DATA PREPARATION.\nArguments:\n{pd.Series(vars(args))}", end='\n\n')

# run data preparation pipeline
e2e_data(args.data)
