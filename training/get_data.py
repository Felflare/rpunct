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
COMPOSITE_ARTICLES_START = 2022


def remove_temp_files(dataset_path, extensions=['npy', 'csv'], traintest=''):
    if traintest != '':
        traintest = '*' + traintest

    # find directory and remove all temporary .npy and .csv files
    for extension in extensions:
        for p in pathlib.Path(dataset_path).glob(f"{traintest}*." + extension):
            p.unlink()


def collate_news_articles(start_date, end_date, summary_or_body='body', train_split=0.9, composite=False, dataset_path=PATH):
    # create directory for data storage
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    # get news article data
    print(f"\n> Assembling news article {summary_or_body[:-1]}ies:")
    news_datasets = [f'news_{date}.jsonl' for date in range(start_date, end_date + 1)]
    articles = []

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'source_articles_news', dataset_json)

        with open(json_path, 'r') as fp:
            for line in fp:
                obj = json.loads(line)
                articles.append(obj[summary_or_body])
                del obj

    # train-test split
    random.seed(42)
    random.shuffle(articles)
    split = math.ceil(train_split * len(articles))
    train = articles[:split]
    test = articles[split:]

    print(f"\t* Articles in total    : {len(articles)}")
    print(f"\t* Articles in train set: {len(train)}")
    print(f"\t* Articles in test set : {len(test)}")
    del articles

    # save train/test data to csv
    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n', ' ')
    csv_path_train = os.path.join(dataset_path, 'train_news.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(dataset_path, 'test_news.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    return True


def collate_news_transcripts(train_split=0.9, composite=False, dataset_path=PATH):
    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    # input transcripts from json files
    print(f"\n> Assembling news transcripts:")
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

    # train-test split
    random.seed(42)
    random.shuffle(transcripts)
    split = math.ceil(train_split * len(transcripts))
    train = transcripts[:split]
    test = transcripts[split:]

    print(f"\t* {train_split:.1f} : {1-train_split:.1f} data split")
    print(f"\t* Speaker segments in total    : {len(transcripts)}")
    print(f"\t* Speaker segments in train set: {len(train)}")
    print(f"\t* Speaker segments in test set : {len(test)}")
    del transcripts

    # save train/test data to csv
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n',' ')
    csv_path_train = os.path.join(dataset_path, 'train_transcripts.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(dataset_path, 'test_transcripts.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    return True


def collate_subtitles(train_split=0.9, composite=False, dataset_path=PATH):
    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    # input transcripts from json files
    print(f"\n> Assembling subtitles data:")

    news_subs_path = os.path.join(PATH, 'source_subtitles_news')
    news_subs_datasets = [os.path.join(news_subs_path, file) for file in os.listdir(news_subs_path) if file.endswith('.json')]
    other_subs_path = os.path.join(PATH, 'source_subtitles_other')
    other_subs_datasets = [os.path.join(other_subs_path, file) for file in os.listdir(other_subs_path) if file.endswith('.json')]
    subs_datasets = news_subs_datasets + other_subs_datasets
    transcripts = np.empty(shape=(0), dtype=object)

    print(f"\t* News subtitle transcripts : {len(news_subs_datasets)}")
    print(f"\t* Other subtitle transcripts: {len(other_subs_datasets)}")
    print(f"\t* All subtitle transcripts  : {len(subs_datasets)}")

    with tqdm(subs_datasets) as D:
        D.set_description("        * Reading subtitle data")
        for json_path in D:
            with open(json_path, 'r') as f:
                obj = json.load(f)

            data = obj["results"]["transcripts"][0]["transcript"]
            transcripts = np.append(transcripts, data)

            del obj
            del data

    # train-test split
    random.seed(42)
    random.shuffle(transcripts)
    split = math.ceil(train_split * len(transcripts))
    train = transcripts[:split]
    test = transcripts[split:]

    print(f"\t* {train_split:.1f} : {1-train_split:.1f} data split")
    print(f"\t* Transcripts in train set: {len(train)} / {len(transcripts)}")
    print(f"\t* Transcripts in test set : {len(test)} / {len(transcripts)}")
    del transcripts

    # save train/test data to csv
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n',' ')
    csv_path_train = os.path.join(dataset_path, 'train_subtitles.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(dataset_path, 'test_subtitles.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    return True


def download_reviews(dataset_path=PATH):
    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    # download yelp reviews data from tensorflow_datasets
    train, test = tfds.load('yelp_polarity_reviews', split=['train', 'test'], shuffle_files=True)

    # train-test split
    train = tfds.as_dataframe(train)
    train['text'] = train['text'].str.decode("utf-8")

    test = tfds.as_dataframe(test)
    test['text'] = test['text'].str.decode("utf-8")

    # filter to only positive examples
    train = train[train['label'] == 1].reset_index(drop=True)
    train = train.drop(columns=['label'])
    test = test[test['label'] == 1].reset_index(drop=True)
    test = test.drop(columns=['label'])

    # save train/test data to csv
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    csv_path_train = os.path.join(dataset_path, 'train_reviews.csv')
    train.to_csv(csv_path_train, index=False)

    csv_path_test = os.path.join(dataset_path, 'test_reviews.csv')
    test.to_csv(csv_path_test, index=False)

    return True


def create_composite_dataset(dataset_names, train_split=0.9, balance=None, dataset_dir=PATH):
    # remove pre-existing data files
    remove_temp_files(dataset_dir, extensions=['npy', 'csv', 'txt'])

    # create a collection of all individual datasets needed to construct composite datasets
    all_datasets = []

    for name in dataset_names:
        # collect dataset from file
        if name == 'news-articles':
            # collect articles part of composite dataset (from JSONL files)
            collate_news_articles(COMPOSITE_ARTICLES_START, 2022, train_split=1.0, composite=True, dataset_path=dataset_dir)
            dataset_path = os.path.join(dataset_dir, 'train_news.csv')

        elif name == 'news-transcripts':
            # collect transcripts part of dataset
            collate_news_transcripts(train_split=train_split, composite=True, dataset_path=dataset_dir)
            dataset_path = os.path.join(dataset_dir, 'train_transcripts.csv')

            # construct testing data from transcripts dataset
            test_dataset_path_input = os.path.join(dataset_dir, 'test_transcripts.csv')
            transcripts_test = pd.read_csv(test_dataset_path_input)

            # save test data file
            test_dataset_path_output = os.path.join(dataset_dir, 'test_composite.csv')

            if os.path.isfile(test_dataset_path_output):
                other_test = pd.read_csv(test_dataset_path_output)
                transcripts_test = pd.concat([transcripts_test, other_test], axis=1)
                transcripts_test = transcripts_test.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

            transcripts_test.dropna(inplace=True)
            transcripts_test.reset_index(drop=True, inplace=True)
            transcripts_test.to_csv(test_dataset_path_output, index=False)
            del transcripts_test

        elif name == 'subtitles':
            # collect subtitles part of dataset
            collate_subtitles(train_split=train_split, composite=True, dataset_path=dataset_dir)
            dataset_path = os.path.join(dataset_dir, 'train_subtitles.csv')

            # split out test dataset
            test_dataset_path_input = os.path.join(dataset_dir, 'test_subtitles.csv')
            subtitles_test = pd.read_csv(test_dataset_path_input)

            # save test data file
            test_dataset_path_output = os.path.join(dataset_dir, 'test_composite.csv')

            if os.path.isfile(test_dataset_path_output):
                other_test = pd.read_csv(test_dataset_path_output)
                subtitles_test = pd.concat([subtitles_test, other_test], axis=1)
                subtitles_test = subtitles_test.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

            subtitles_test.dropna(inplace=True)
            subtitles_test.reset_index(drop=True, inplace=True)
            subtitles_test.to_csv(test_dataset_path_output, index=False)
            del subtitles_test

        else:
            raise ValueError("Composite dataset cannot be built with unknown source datasets.")

        # format dataset
        dataset = pd.read_csv(dataset_path)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        # add to dataset collection
        all_datasets.append(dataset.copy())
        del dataset

    # combine two news datasets together in proportion denoted by `balance`
    if balance != 'o':
        balance = list(map(int, balance.split(':')))
        if len(balance) == len(all_datasets):
            print("\n> Proportioning datasets:")
            print(f"\t* Approx. ratio of dataset sizes: {balance}")

            # clip all datasets to the size of the smallest (so proportions are relative to that)
            min_dataset_len = min(map(len, all_datasets))
            all_datasets = [d[:min_dataset_len] for d in all_datasets]

            balace_total = sum(balance)
            data_total = sum(map(len, all_datasets))
            proportions = [round(data_total / balace_total) * b for b in balance]
            proportioned_datasets = []

            for d, p in zip(all_datasets, proportions):
                p = min(p, len(d))
                d = d[:p]
                proportioned_datasets.append(d)

            all_datasets = proportioned_datasets.copy()
            del proportioned_datasets
            print(f"\t* Proportioned dataset sizes: {list(map(len, all_datasets))}")

    # combine separate data of varying types into single composite dataset
    composite_data = pd.concat(all_datasets, ignore_index=True)
    del all_datasets
    composite_data = composite_data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle samples

    # save composite dataset to csv file
    csv_path_train = os.path.join(dataset_dir, 'train_composite.csv')
    composite_data.to_csv(csv_path_train, index=False)
    del composite_data

    return True
