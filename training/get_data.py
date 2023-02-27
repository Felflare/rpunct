import os
import json
import math
import random
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow_datasets as tfds

PATH = './training/datasets/'
COMPOSITE_ARTICLES_START = 2020


def check_data_exists(data_source='news-transcripts', train_or_test='train'):
    """
    Error checking function to ensure training/testing dataset has been constructed before model training/testing commences.

    Args:
        - data_type (str): the name of the directory where training/testing data files are expected to be located.
        - train_or_test (str): specifier of whether to check for training or testing dataset.
    """
    data_path = os.path.join(PATH, data_source, f"rpunct_{train_or_test}_set.txt")
    data_files_exist = os.path.isfile(data_path)
    print("\n> Checking dataset exists:")
    print(f"\t* Required data file: {data_path}")
    print(f"\t* Found: {data_files_exist}")

    return data_files_exist


def remove_temp_files(directory, extensions, traintest=''):
    """
    Remove temporary intermediate files generated during the data preparation process.

    Args:
        - directory (Path): location of directory to remove files from (relative to top-level `rpunct` directory).
        - extensions (list): extensions of file types to remove.
        - traintest (string): specify `train` or `test` if you only want to remove files related to a specific dataset.
    """
    if traintest == '':
        pattern = '*.'
    else:
        pattern = '*' + traintest + '*.'

    for extension in extensions:
        files = pathlib.Path(directory).glob(pattern + extension)
        for p in files:
            p.unlink()


def collate_news_articles(start_date, end_date, summary_or_body='body', train_split=0.9, output_directory=PATH):
    """
    Loads in JSON data from BBC News articles (stored within the source directory `source_articles_news`) and combine into a single dataset.
    Then splits data into training and testing datasets and save these as individual CSV files.

    Args:
        - start_date (int): start year of date range over which to gather BBC News articles (2014-22).
        - end_date (int): end year of date range over which to gather BBC News articles (2014-22).
        - summary_or_body (str): toggle between extracting article summaries and article bodies.
        - train_split (float): the proportion of articles to use as training data (decimal between 0-1).
        - output_directory (Path): location of output directory to store output CSV files (relative to top-level `rpunct` directory).
    """
    print(f"\n> Assembling news article {summary_or_body[:-1]}ies:")

    # Create directory for data storage
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Remove pre-existing data files from previous iterations
    remove_temp_files(output_directory, extensions=['npy', 'txt'])

    # Input news article data from JSONL files
    news_datasets = [f'news_{date}.jsonl' for date in range(start_date, end_date + 1)]
    articles = []

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'source_articles_news', dataset_json)

        with open(json_path, 'r') as fp:
            for line in fp:
                obj = json.loads(line)
                articles.append(obj[summary_or_body])
                del obj

    # Train-test split
    random.seed(42)
    random.shuffle(articles)
    split = math.ceil(train_split * len(articles))
    train = articles[:split]
    test = articles[split:]

    print(f"\t* Data split            : {train_split:.2f}:{1 - train_split:.2f}")
    print(f"\t* Articles in total     : {len(articles)}")
    print(f"\t* Articles in train set : {len(train)}")
    print(f"\t* Articles in test set  : {len(test)}")
    del articles

    # Save train/test data to csv
    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n', ' ')
    csv_path_train = os.path.join(output_directory, 'train_news.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(output_directory, 'test_news.csv')
    test.to_csv(csv_path_test, index=False)
    del test


def collate_news_transcripts(train_split=0.9, output_directory=PATH):
    """
    Loads in JSON data from BBC News transcripts (stored within the source directory `source_transcripts_news`) and combine into a single dataset.
    Then splits data into training and testing datasets and save these as individual CSV files.

    Args:
        - train_split (float): the proportion of transcripts to use as training data (decimal between 0-1).
        - output_directory (Path): location of output directory to store output CSV files (relative to top-level `rpunct` directory).
    """
    print(f"\n> Assembling news transcripts:")

    # Remove pre-existing data files from previous iterations
    remove_temp_files(output_directory, extensions=['npy', 'txt'])

    # Input transcripts from JSON files
    news_datasets = ['transcripts_2014-17.json', 'transcripts_2020.json']
    transcripts = np.empty(shape=(0), dtype=object)

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'source_transcripts_news', dataset_json)

        with open(json_path, 'r') as f:
            obj = json.load(f)

        speaker_segments = []
        for data in obj["Transcripts"]:
            speaker_segments.extend(data["Items"])

        transcripts = np.append(transcripts, speaker_segments)

        del obj
        del speaker_segments

    # Train-test split
    random.seed(42)
    random.shuffle(transcripts)
    split = math.ceil(train_split * len(transcripts))
    train = transcripts[:split]
    test = transcripts[split:]

    print(f"\t* Data split                    : {train_split:.2f}:{1 - train_split:.2f}")
    print(f"\t* Speaker segments in total     : {len(transcripts)}")
    print(f"\t* Speaker segments in train set : {len(train)}")
    print(f"\t* Speaker segments in test set  : {len(test)}")
    del transcripts

    # Save train/test data to csv
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n',' ')
    csv_path_train = os.path.join(output_directory, 'train_transcripts.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(output_directory, 'test_transcripts.csv')
    test.to_csv(csv_path_test, index=False)
    del test


def collate_subtitles(train_split=0.9, output_directory=PATH):
    """
    Loads in JSON data from BBC subtitles (stored within the source directories `source_subtitles_news` & `source_subtitles_other`) and combine into a single dataset.
    Then splits data into training and testing datasets and save these as individual CSV files.

    Args:
        - train_split (float): the proportion of subtitle transcripts to use as training data (decimal between 0-1).
        - output_directory (Path): location of output directory to store output CSV files (relative to top-level `rpunct` directory).
    """
    print(f"\n> Assembling subtitles data:")

    # Remove pre-existing data files from previous iterations
    remove_temp_files(output_directory, extensions=['npy', 'txt'])

    # Input subtitle transcripts from JSON files
    news_subs_path = os.path.join(PATH, 'source_subtitles_news')
    news_subs_datasets = [os.path.join(news_subs_path, file) for file in os.listdir(news_subs_path) if file.endswith('.json')]
    other_subs_path = os.path.join(PATH, 'source_subtitles_other')
    other_subs_datasets = [os.path.join(other_subs_path, file) for file in os.listdir(other_subs_path) if file.endswith('.json')]
    subs_datasets = news_subs_datasets + other_subs_datasets
    transcripts = np.empty(shape=(0), dtype=object)
    print(f"\t* Data split {' ' * 22} : {train_split:.2f}:{1 - train_split:.2f}")

    with tqdm(subs_datasets) as D:
        D.set_description(f"{' ' * 7} * Extracting subtitles data {' ' * 8}")
        for json_path in D:
            with open(json_path, 'r') as f:
                obj = json.load(f)

            data = obj["results"]["transcripts"][0]["transcript"]
            transcripts = np.append(transcripts, data)

            del obj
            del data

    # Train-test split
    random.seed(42)
    random.shuffle(transcripts)
    split = math.ceil(train_split * len(transcripts))
    train = transcripts[:split]
    test = transcripts[split:]
    del transcripts

    print(f"\t* Subtitle transcripts in train set : {len(train)}")
    print(f"\t* Subtitle transcripts in test set  : {len(test)}")

    # Save train/test data to csv
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n', ' ')
    csv_path_train = os.path.join(output_directory, 'train_subtitles.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(output_directory, 'test_subtitles.csv')
    test.to_csv(csv_path_test, index=False)
    del test


def download_reviews(output_directory=PATH):
    """
    Loads in Yelp reviews data from Tensorflow Datasets.
    Then splits data into training and testing datasets and save these as individual CSV files.

    Args:
        - output_directory (Path): location of output directory to store output CSV files (relative to top-level `rpunct` directory).
    """
    print(f"\n> Assembling reviews data:")

    # Remove pre-existing data files from previous iterations
    remove_temp_files(output_directory, extensions=['npy', 'txt'])

    # Download Yelp reviews data from `tensorflow_datasets`
    train, test = tfds.load('yelp_polarity_reviews', split=['train', 'test'], shuffle_files=True)

    # Train-test split
    train = tfds.as_dataframe(train)
    train['text'] = train['text'].str.decode("utf-8")

    test = tfds.as_dataframe(test)
    test['text'] = test['text'].str.decode("utf-8")

    # Filter to only positive instances
    train = train[train['label'] == 1].reset_index(drop=True)
    train = train.drop(columns=['label'])
    test = test[test['label'] == 1].reset_index(drop=True)
    test = test.drop(columns=['label'])

    # Save train/test data to csv
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    csv_path_train = os.path.join(output_directory, 'train_reviews.csv')
    train.to_csv(csv_path_train, index=False)

    csv_path_test = os.path.join(output_directory, 'test_reviews.csv')
    test.to_csv(csv_path_test, index=False)


def create_composite_dataset(dataset_names, train_split=0.9, balance='o', output_directory=PATH):
    """
    Loads in and combines data from multiple sources (using the above collation functions) into a single composite dataset.
    Then splits composite data into training and testing datasets and save these as individual CSV files.

    Args:
        - dataset_names (list): string name tags of each data type to include in the composite dataset.
        - train_split (float): the proportion of subtitle transcripts to use as training data (decimal between 0-1).
        - balance (str): specifier to clip dataset sizes to be included in composite dataset in a given ratio to each other (in form `x:y`)
        - output_directory (Path): location of output directory to store output CSV files (relative to top-level `rpunct` directory).
    """
    # Remove pre-existing data files
    remove_temp_files(output_directory, extensions=['npy', 'csv', 'txt'])

    # Gather a collection of all individual datasets needed to construct composite datasets from
    train_datasets = []
    test_datasets = []

    for name in dataset_names:
        # Collect each dataset from local files
        if name == 'news-articles':
            # Collect transcripts part of dataset (only used for training not testing)
            collate_news_articles(COMPOSITE_ARTICLES_START, 2022, train_split=1.0, output_directory=output_directory)
            dataset_path = os.path.join(output_directory, 'train_news.csv')

        elif name == 'news-transcripts':
            # Collect transcripts part of dataset
            collate_news_transcripts(train_split=train_split, output_directory=output_directory)
            dataset_path = os.path.join(output_directory, 'train_transcripts.csv')
            test_data_path = os.path.join(output_directory, 'test_transcripts.csv')

            # Save test dataset
            test_data = pd.read_csv(test_data_path)
            test_data.dropna(inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            test_datasets.append(test_data.copy())
            del test_data

        elif name == 'subtitles':
            # Collect subtitles part of dataset
            collate_subtitles(train_split=train_split, output_directory=output_directory)
            dataset_path = os.path.join(output_directory, 'train_subtitles.csv')
            test_data_path = os.path.join(output_directory, 'test_subtitles.csv')

            # Save test dataset
            test_data = pd.read_csv(test_data_path)
            test_data.dropna(inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            test_datasets.append(test_data.copy())
            del test_data

        else:
            raise ValueError("Composite dataset cannot be built with unknown source datasets.")

        # Format dataset
        dataset = pd.read_csv(dataset_path)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        # Add to dataset collection
        train_datasets.append(dataset.copy())
        del dataset

    # Combine multiple datasets together in a specific proportion denoted by `balance`
    print("\n> Proportioning data:")
    if balance == 'o':
        print(f"\t* Using original sizes : {list(map(len, train_datasets))}")
    else:
        balance = list(map(int, balance.split(':')))
        if len(balance) == len(train_datasets):
            print(f"\t* Desired ratio of dataset sizes : {balance}")

            # Clip all datasets to the size of the smallest (so proportions are relative to that)
            clipped_dataset_len = min(map(len, train_datasets))
            train_datasets = [d[:clipped_dataset_len] for d in train_datasets]

            balance_max = max(balance)
            proportions = [round((b / balance_max) * clipped_dataset_len) for b in balance]
            proportioned_datasets = []

            for d, p in zip(train_datasets, proportions):
                p = min(p, len(d))
                d = d[:p]
                proportioned_datasets.append(d)

            train_datasets = proportioned_datasets
            print(f"\t* Proportioned dataset sizes     : {list(map(len, train_datasets))}")

    # Combine separate training data of varying types into single composite dataset
    composite_data = pd.concat(train_datasets, ignore_index=True)
    del train_datasets
    composite_data = composite_data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle samples

    # Save composite training dataset to csv file
    csv_path_train = os.path.join(output_directory, 'train_composite.csv')
    composite_data.to_csv(csv_path_train, index=False)
    del composite_data

    # Similarly combine test data
    composite_test = pd.concat(test_datasets, ignore_index=True)
    del test_datasets
    composite_test = composite_test.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle samples

    # Save composite training dataset to csv file
    csv_path_test = os.path.join(output_directory, 'test_composite.csv')
    composite_test.to_csv(csv_path_test, index=False)
    del composite_test
