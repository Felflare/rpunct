import pandas as pd
import argparse
from training.test import e2e_test

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
    help="Which models would you like to test? Please provide the path to the directory holding any model's parameterisation."
)

parser.add_argument(
    '-d',
    '--data',
    metavar='DATA',
    type=str,
    choices=['news', 'reviews'],
    default='news',
    help='What dataset should be used to test the model: BBC News or Yelp reviews?'
)

parser.add_argument(
    '-c',
    '--cuda',
    action='store_true',
    default=False,
    help="Would you like to run testing on a GPU using CUDA?"
)

parser.add_argument(
    '-s',
    '--statistics',
    action='store_true',
    default=False,
    help="Would you like to print statistics about the test dataset you have selected?"
)

# parse arguments
args = parser.parse_args()
print(pd.DataFrame(vars(args)))

e2e_test(
    args.models,
    data_type=args.data,
    use_cuda=args.cuda,
    print_stats=args.statistics
)
