import argparse
import pandas as pd
from training.test import e2e_test
from training.train import e2e_train
from training.prep_data import e2e_data, check_data_exists
from rpunct.punctuate import run_rpunct


# Parser
parser = argparse.ArgumentParser(description='Global run file to execute dataset preparation, model training, and testing.')
subparsers = parser.add_subparsers(help="Specify which stage of the ML process to be executed: data preparation (`data`), training (`train`), testing (`test`), or inference (`punct`).", dest="stage")
data_parser = subparsers.add_parser('data', help='Execute data preparation process.')
train_parser = subparsers.add_parser('train', help='Execute model training process.')
test_parser = subparsers.add_parser('test', help='Execute model testing process.')
punct_parser = subparsers.add_parser('punct', help='Run rpunct on a given input of plaintext.')

# Data subparsers
data_source_subparsers = data_parser.add_subparsers(help="Specify type of data to be prepared.", dest="data")
reviews_data_subparser = data_source_subparsers.add_parser('reviews', help='Yelp reviews dataset.')
news_data_subparser = data_source_subparsers.add_parser('news', help='BBC News articles dataset.')
transcripts_data_subparser = data_source_subparsers.add_parser('news-transcripts', help='BBC News transcripts dataset.')
subtitles_data_subparser = data_source_subparsers.add_parser('subtitles', help='BBC subtitles (all genres) dataset.')
composite_data_subparser = data_source_subparsers.add_parser('composite', help='Composite dataset including data from multiple sources (e.g. articles and transcripts).')

# Data arguments
data_parser.add_argument(
    '-sp',
    '--split',
    metavar='TRAIN:TEST',
    action='store',
    type=str,
    default='90:10',
    help="Specify the train-test split to be implemented (TRAIN perc. of data for training, TEST for testing) - default is 90:10."
)

news_data_subparser.add_argument(
    '-s',
    '--start',
    metavar='YEAR',
    type=str,
    choices=[str(year) for year in range(2014, 2023)],
    default='2014',
    help="Specify the start year of the range of news articles you want to input as the dataset - default is 2014."
)

news_data_subparser.add_argument(
    '-e',
    '--end',
    metavar='YEAR',
    type=str,
    choices=[str(year) for year in range(2014, 2023)],
    default='2022',
    help="Specify the end year of the range of news articles you want to input as the dataset - default is 2022."
)

news_data_subparser.add_argument(
    '-sm',
    '--sum',
    action='store_true',
    default=False,
    help="Toggle between BBC News article summaries and bodies - default is bodies."
)

composite_data_subparser.add_argument(
    '-i',
    '--include',
    metavar='DATASET',
    action='store',
    nargs='+',
    type=str,
    default=['news-articles', 'news-transcripts'],
    help="Specify the 2+ data sources to include in the composite dataset - default is news articles and transcripts."
)

composite_data_subparser.add_argument(
    '-d',
    '--distinct',
    action='store_true',
    default=False,
    help="Toggle between an integrated composite dataset and distinct datasets for pre-training / fine-tuning - default is integrated."
)

composite_data_subparser.add_argument(
    '-b',
    '--databalance',
    metavar='PROP_1:PROP_2...',
    action='store',
    type=str,
    default='o',
    help="Specify the proportion by which to combine the multiple datasets - default is their original lengths."
)


# Training arguments
train_parser.add_argument(
    '-d',
    '--data',
    metavar='DATA',
    type=str,
    choices=['reviews', 'news-summaries', 'composite', 'composite-news-int', 'composite-news-dist', 'news-transcripts', 'subtitles'].extend([f'news-{start}-{end}' for start in range(2014, 2023) for end in range(2014, 2023)]),
    default='news-2014-2022',
    help="Specify the (path to the) dataset to be used to test the model: BBC News (`news-startyr-endyr`) or Yelp reviews (`reviews`) - default is BBC News 2014-2022."
)

train_parser.add_argument(
    '-e',
    '--epochs',
    metavar='EPOCHS',
    action='store',
    type=int,
    default=3,
    help="Specify the number of epochs over which to run training - default is 3."
)

train_parser.add_argument(
    '-v',
    '--val',
    action='store_true',
    default=False,
    help="Toggle validation, where the model is evaluated every 5000 steps during training - default is off."
)

train_parser.add_argument(
    '-ft',
    '--finetune',
    action='store_true',
    default=False,
    help="Toggle on fine-tuning training round after initial pre-training stage - default is off."
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
    choices=['reviews', 'news-summaries', 'news-sum', 'composite-news-int', 'composite-news-dist', 'comp-news', 'news-transcripts', 'news-trans', 'subtitles'].extend([f'news-{start}-{end}' for start in range(2014, 2023) for end in range(2014, 2023)]),
    default='news-2014-2022',
    help="Specify the (path to the) dataset to be used to test the model: BBC News (`news-startyr-endyr`) or Yelp reviews (`reviews`) - default is BBC News 2014-2022."
)

test_parser.add_argument(
    '-o',
    '--output',
    metavar='OUT_PNG',
    type=str,
    default='model_performance.png',
    help="Specify the output file to save the performance comparision plot - default is `model_performance`."
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


if __name__ == "__main__":
    # Parse these arguments
    args = parser.parse_args()
    print("\n> Arguments:", end='\n\n')
    print(pd.Series(vars(args)))

    # if calling for inference, run punctuate.py function
    if args.stage == 'punct':
        # generate instance of rpunct model and run text through it
        run_rpunct(
            use_cuda=args.cuda,
            input_txt=args.input,
            output_txt=args.output,
            model_location=args.model
        )

    else:
        # Run the pipeline for the ML processing stage selected (data prep, train, test)
        if args.stage == 'data':
            # error checking
            if args.data == 'news':
                if args.end < args.start:
                    raise ValueError("End year of news data range must not be earlier than start year.")

            elif args.data == 'composite':
                if len(args.include) < 2:
                    raise ValueError(f"If specifying a composite dataset, at least two data sources must be specified (to merge together). You only specified {len(args.datasets)}.")

            # run data preparation pipeline
            if args.data == 'news':
                e2e_data(
                    data_type=args.data,
                    start_year=args.start,
                    end_year=args.end,
                    summaries=args.sum,
                    tt_split=args.split
                )

            elif args.data == 'composite':
                e2e_data(
                    data_type=args.data,
                    tt_split=args.split,
                    composite_datasets_list=args.include,
                    composite_data_distinctness=args.distinct,
                    dataset_balance=args.databalance
                )

            else:  # currently (args.data == 'reviews') and (args.data == 'news-transcripts')
                e2e_data(
                    data_type=args.data,
                    tt_split=args.split
                )

        elif args.stage in ['train', 'test']:
            # run data preparation pipeline if dataset does not exist
            if args.data[:7] == 'news-20':  # articles between two dates
                data_type, data_start, data_end = args.data.split('-')
                summaries = False
            elif args.data[:8] == 'news-sum':  # summaries
                data_type, summaries, data_start, data_end = 'news', True, '', ''
            else:  # transcripts, composite, etc.
                data_type, summaries, data_start, data_end = args.data, False, '', ''

            if args.stage == 'train' and args.data == 'composite-news-dist':
                args.finetune = True
            else:
                args.finetune = False

            dataset_exists = check_data_exists(
                data_type=data_type,
                train_or_test=args.stage,
                start_date=data_start,
                end_date=data_end,
                summaries=summaries,
                finetuning=args.finetune
            )

            if not dataset_exists:
                e2e_data(
                    data_type=data_type,
                    start_year=data_start,
                    end_year=data_end,
                    summaries=summaries,
                    composite_datasets_list=['news-articles', 'news-transcripts'],
                    composite_data_distinctness=args.finetune
                )

            if args.stage == 'train':
                # run pipeline to build and train language model
                e2e_train(
                    data_source=args.data,
                    use_cuda=args.cuda,
                    validation=args.val,
                    dataset_stats=args.stats,
                    training_plot=args.plot,
                    epochs=args.epochs,
                    conduct_fine_tuning=args.finetune
                )
            else:  # args.stage == 'test'
                # run model testing pipeline
                e2e_test(
                    args.models,
                    data_source=args.data,
                    use_cuda=args.cuda,
                    print_stats=args.stats,
                    output_file=args.output
                )
