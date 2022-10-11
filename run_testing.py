import pandas as pd
import argparse
from training.test import e2e_test
from training.prep_data import e2e_data, check_data_exists


# parser
parser = argparse.ArgumentParser(description='Test the performance of rpunct model and/or compare it to another parameterisation.')

# arguments
parser.add_argument(
    '-m',
    '--models',
    metavar='MODEL',
    action='store',
    nargs='+',
    type=str,
    help="Specify the model(s) to be tested. Please provide the path to the directory holding any model's parameterisation."
)

parser.add_argument(
    '-d',
    '--data',
    metavar='DATA',
    type=str,
    choices=['news', 'reviews'],
    default='news',
    help="Specify the dataset to be used to test the model: BBC News or Yelp reviews."
)

parser.add_argument(
    '-c',
    '--cuda',
    action='store_true',
    default=False,
    help="Toggle between training on a GPU using CUDA or on the CPU."
)

parser.add_argument(
    '-s',
    '--statistics',
    action='store_true',
    default=False,
    help="Print label distribution statistics about the test dataset."
)

# parse arguments
args = parser.parse_args()
print(f"RUNNING TESTING.\nArguments:")
print(pd.DataFrame(vars(args)), end='\n\n')

# run data preparation pipeline if training dataset does not exist
dataset_exists = check_data_exists(data_type=args.data, train_or_test='test')
if not dataset_exists:
    e2e_data(args.data)

# run model testing pipeline
e2e_test(
    args.models,
    data_type=args.data,
    use_cuda=args.cuda,
    print_stats=args.stats
)
