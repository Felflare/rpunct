import os
import pathlib
import argparse
import pandas as pd
from training.test import e2e_test
from training.train import e2e_train
from training.prep_data import e2e_data, check_data_exists
from rpunct.punctuate import run_rpunct


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Global run file to execute dataset preparation, model training, and testing.')
    subparsers = parser.add_subparsers(help="Specify which stage of the ML process to be executed: data preparation (`data`), training (`train`), testing (`test`), or inference (`punct`).", dest="stage")
    data_parser = subparsers.add_parser('data', help='Execute data preparation process.')
    train_parser = subparsers.add_parser('train', help='Execute model training process.')
    test_parser = subparsers.add_parser('test', help='Execute model testing process.')
    punct_parser = subparsers.add_parser('punct', help='Run rpunct on a given input of plaintext.')

    # Data arguments
    data_parser.add_argument(
        '-d',
        '--data',
        metavar='DATA',
        type=str,
        choices=['news', 'reviews'],
        default='news',
        help="Specify the dataset to be used to test the model: BBC News (`news`) or Yelp reviews (`reviews`) - default is BBC News."
    )

    # Training arguments
    train_parser.add_argument(
        '-d',
        '--data',
        metavar='DATA',
        type=str,
        choices=['news', 'reviews'],
        default='news',
        help="Specify the dataset to be used to test the model: BBC News (`news`) or Yelp reviews (`reviews`) - default is BBC News."
    )

    train_parser.add_argument(
        '-v',
        '--val',
        action='store_true',
        default=False,
        help="Toggle validation, where the model is evaluated every 5000 steps during training - default is off."
    )

    train_parser.add_argument(
        '-c',
        '--cuda',
        action='store_true',
        default=False,
        help="Toggle between training on a GPU using CUDA or on the CPU - default is CPU."
    )

    train_parser.add_argument(
        '-p',
        '--plot',
        action='store_true',
        default=False,
        help="Output a plot of the convergence of the training/validation loss function - default is off."
    )

    train_parser.add_argument(
        '-s',
        '--stats',
        action='store_true',
        default=False,
        help="Print label distribution statistics about the test dataset - default hides stats."
    )

    # Testing arguments
    test_parser.add_argument(
        'models',
        metavar='MODEL',
        action='store',
        nargs='+',
        type=str,
        help="Specify the model(s) to be tested. Please provide the path to the directory holding any model's parameterisation - required."
    )

    test_parser.add_argument(
        '-d',
        '--data',
        metavar='DATA',
        type=str,
        choices=['news', 'reviews'],
        default='news',
        help="Specify the dataset to be used to test the model: BBC News (`news`) or Yelp reviews (`reviews`) - default is BBC News."
    )

    test_parser.add_argument(
        '-c',
        '--cuda',
        action='store_true',
        default=False,
        help="Toggle between training on a GPU using CUDA or on the CPU - default is CPU."
    )

    test_parser.add_argument(
        '-s',
        '--stats',
        action='store_true',
        default=False,
        help="Print label distribution statistics about the test dataset - default hides stats."
    )

    # Punctuate arguments
    punct_parser.add_argument(
        '-m',
        '--model',
        metavar='MODEL',
        action='store',
        type=str,
        default='felflare/bert-restore-punctuation',
        help="Specify the model to be used to generate punctuations (by giving the path to its parameterisation) - default uses `felflare/bert-restore-punctuation`."
    )

    punct_parser.add_argument(
        '-i',
        '--input',
        metavar='IN_TXT',
        type=str,
        default='tests/sample_text.txt',
        help="Specify the path (from the current directory) to the txt file contaning plaintext to be punctuated - default is `tests/sample_text.txt`"
    )

    punct_parser.add_argument(
        '-o',
        '--output',
        metavar='OUT_TXT',
        action='store',
        type=str,
        default=None,
        help="Specify the path to the output file to print punctuated text - default prints to command line."
    )

    punct_parser.add_argument(
        '-c',
        '--cuda',
        action='store_true',
        default=False,
        help="Toggle between training on a GPU using CUDA or on the CPU - default is CPU."
    )

    # Parse these arguments
    args = parser.parse_args()
    print(f"Arguments:\n{pd.Series(vars(args))}")

    # Run the pipeline for the specifc ML processing stage selected
    if args.stage == 'data':
        # run data preparation pipeline
        e2e_data(args.data)

    elif args.stage in ['train', 'test']:
        # run data preparation pipeline if dataset does not exist
        dataset_exists = check_data_exists(data_type=args.data, train_or_test=args.stage)
        if not dataset_exists:
            e2e_data(args.data)

        if args.stage == 'train':
            # run pipeline to build and train language model
            e2e_train(
                data_type=args.data,
                use_cuda=args.cuda,
                validation=args.val,
                dataset_stats=args.stats,
                training_plot=args.plot
            )
        else:  # args.stage == 'test'
            # run model testing pipeline
            e2e_test(
                args.models,
                data_type=args.data,
                use_cuda=args.cuda,
                print_stats=args.stats
            )

    elif args.stage == 'punct':
        # generate instance of rpunct model and run text through it
        run_rpunct(
            use_cuda=args.cuda,
            input_txt=args.input,
            output_txt=args.output,
            model_location=args.model
        )
