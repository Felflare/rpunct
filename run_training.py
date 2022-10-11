import pandas as pd
import argparse
from training.prep_data import e2e_data, check_data_exists
from training.train import e2e_train


# parser
parser = argparse.ArgumentParser(description='Prepare training/testing data, build language model, and train model over training dataset.')

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

parser.add_argument(
    '-v',
    '--val',
    action='store_true',
    default=False,
    help="Toggle validation, where the model is evaluated every 5000 steps during training"
)

parser.add_argument(
    '-c',
    '--cuda',
    action='store_true',
    default=False,
    help="Toggle between training on a GPU using CUDA or on the CPU"
)

parser.add_argument(
    '-s',
    '--stats',
    action='store_true',
    default=False,
    help="Print label distribution statistics about the test dataset."
)

parser.add_argument(
    '-p',
    '--plot',
    action='store_true',
    default=False,
    help="Output a plot of the convergence of the training/validation loss function."
)

# parse arguments
args = parser.parse_args()
print(f"RUNNING TRAINING.\nArguments:\n{pd.Series(vars(args))}", end='\n\n')

# run data preparation pipeline if training dataset does not exist
dataset_exists = check_data_exists(data_type=args.data)
if not dataset_exists:
    e2e_data(args.data)

# run pipeline to build and train language model
e2e_train(
    data_type=args.data,
    use_cuda=args.cuda,
    validation=args.val,
    dataset_stats=args.stats,
    training_plot=args.plot
)
