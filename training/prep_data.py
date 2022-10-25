# -*- coding: utf-8 -*-
# 💾⚙️🔮
__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import re
import json
import math
import random
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow_datasets as tfds

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
PATH = './training/datasets/'
NO_OUTPUT_FILES = 5


def e2e_data(data_type='news', start_year='2014', end_year='2022', summaries=False):
    """
    Full pipeline for compiling and formatting training data from BBC News articles or Yelp reviews
    """
    # generate/collect raw data
    if data_type == 'news':
        # collect data from each JSONL file enumerating all BBC News articles for each year 2014-2022
        print(f"\n> Preparing data from source: BBC News articles")
        dataset_path = collate_news_articles(int(start_year), int(end_year), summaries)
    elif data_type == 'reviews':
        # save training/testing datasets from tensorflow to local csv files
        print("\n> Preparing data from source: Yelp reviews")
        dataset_path = download_reviews()
    elif data_type == 'news-transcripts':
        # extract and process transcripts from JSON files
        print(f"\n> Preparing data from source: BBC News transcripts")
        dataset_path = collate_news_transcripts()
        exit(1)
    else:
        raise ValueError("Unrecognised data source!")

    for key in ['train', 'test']:
        print(f"\n> Generating dataset: {key.upper()}")

        # load in train/test data
        data_split_path = os.path.join(dataset_path, f'{key}_{data_type}.csv')
        data_split = pd.read_csv(data_split_path)
        data_split.dropna(inplace=True)
        data_split.reset_index(drop=True, inplace=True)

        # constuct df of text and labels (punctuation tag per word)
        words_and_labels = create_rpunct_dataset(data_split)
        del data_split

        # split data into chunks for model
        print("\t* Generating data samples")
        output_file = f"{data_type}_{key}"
        create_training_samples(words_and_labels, output_file, file_out_path=dataset_path, num_splits=NO_OUTPUT_FILES, train_or_test=key)
        del words_and_labels

    print("\n> Data generation complete", end='\n\n')


def check_data_exists(data_type='news', train_or_test='train', start_date='2014', end_date='2022', summaries=False):
    # check whether the training data has been created or not yet
    if data_type == 'news':
        if summaries:
            data_dir = os.path.join(PATH, f'news-summaries')
        else:
            data_dir = os.path.join(PATH, f'news-{start_date}-{end_date}')
    else:
        data_dir = os.path.join(PATH, data_type)

    data_file_pattern = f'{data_type}_{train_or_test}_*.npy'
    dataset_paths = list(pathlib.Path(data_dir).glob(data_file_pattern))
    data_files_exist = len(dataset_paths) == NO_OUTPUT_FILES
    print(f"\n> Required data files found at '{data_dir}/{data_file_pattern}'? : {data_files_exist}")

    return data_files_exist


def collate_news_articles(start_date, end_date, summaries):
    if summaries:
        summary_or_body = 'summary'
    else:
        summary_or_body = 'body'

    print(f"\n> Assembling news article {summary_or_body[:-1]}ies:")
    news_datasets = [f'news_{date}.jsonl' for date in range(start_date, end_date + 1)]
    articles = []

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'news_data_2014-22/', dataset_json)

        with open(json_path, 'r') as fp:
            for line in fp:
                obj = json.loads(line)
                articles.append(obj[summary_or_body])
                del obj

    # train-test split
    random.shuffle(articles)
    split = math.ceil(0.9 * len(articles))
    train = articles[:split]
    test = articles[split:]
    print(f"\t* Articles in total    : {len(articles)}")
    print(f"\t* Articles in train set: {len(train)}")
    print(f"\t* Articles in test set : {len(test)}")
    del articles

    # save train/test data to csv
    if summaries:
        dataset_path = os.path.join(PATH, f'news-summaries')
    else:
        dataset_path = os.path.join(PATH, f'news-{start_date}-{end_date}')

    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    csv_path_train = os.path.join(dataset_path, 'train_news.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    csv_path_test = os.path.join(dataset_path, 'test_news.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    return dataset_path


def collate_news_transcripts():
    # input transcripts from json files
    print(f"\n> Assembling news transcripts:")
    news_datasets = ['transcripts_2014-17.json', 'transcripts_2020.json']
    articles = np.empty(shape=(0), dtype=object)

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'news_transcripts_2017-20/', dataset_json)

        with open(json_path, 'r') as f:
            obj = json.load(f)

        transcriptions = pd.DataFrame(obj["Transcripts"])
        speaker_segments = np.concatenate(transcriptions["Items"]).flat
        articles = np.append(articles, speaker_segments)

        del obj
        del transcriptions
        del speaker_segments

    # train-test split
    random.shuffle(articles)
    split = math.ceil(0.9 * len(articles))
    train = articles[:split]
    test = articles[split:]
    print(f"\t* Speaker segments in total    : {len(articles)}")
    print(f"\t* Speaker segments in train set: {len(train)}")
    print(f"\t* Speaker segments in test set : {len(test)}")
    del articles

    # save train/test data to csv
    dataset_path = os.path.join(PATH, f'news-transcripts')
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    csv_path_train = os.path.join(dataset_path, 'train_transcripts.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    csv_path_test = os.path.join(dataset_path, 'test_transcripts.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    return dataset_path


def download_reviews():
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
    dataset_path = os.path.join(PATH, 'reviews')
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    csv_path_train = os.path.join(dataset_path, 'train_reviews.csv')
    train.to_csv(csv_path_train, index=False)

    csv_path_test = os.path.join(dataset_path, 'test_reviews.csv')
    test.to_csv(csv_path_test, index=False)

    return dataset_path


def create_rpunct_dataset(df):
    # constuct df of text and labels (punctuation tag per word)
    all_records = []
    with tqdm(df['text']) as T:
        for article in T:
            T.set_description("        * Labelling data instances")
            records = create_record(article)  # create a list enumerating each word in a single article and its label: [...{id, word, label}...]
            all_records.extend(records)
            del records

    # output the list of all {word, label} dicts
    return all_records


def create_record(row):
    """
    Create labels for Punctuation Restoration task for each token.
    """
    pattern = re.compile("[\W_]+")
    new_obs = []

    # convert string of text (from row) into a list of words
    # row: str -> observation: list[str]
    observation = row.replace('\\n', ' ').split()

    # remove punctuation of each word, and label it with a tag representing what punctuation it did have
    for obs in observation:
        text_obs = obs.lower()  # set lowercase
        text_obs = pattern.sub('', text_obs)  # remove any non-alphanumeric characters

        # if word is the empty string, skip over this one
        if not text_obs:
            continue

        # if there is a punctuation mark after the word, add it to the label
        if not obs[-1].isalnum():
            new_lab = obs[-1]
        else:
            new_lab = "O"  # `O` => no punctuation

        # if the word is capitalised, add it to the label
        if obs[0].isupper():
            new_lab += "U"  # `xU` => capitalised
        else:
            new_lab += "O"  # `xO` => lowercase

        # add the word and its label to the dataset
        new_obs.append({'sentence_id': 0, 'words': text_obs, 'labels': new_lab})

        del text_obs
        del new_lab

    return new_obs


def create_training_samples(all_records, file_out_nm='train_data', file_out_path=PATH, num_splits=5, train_or_test='train'):
    """
    Given a looong list of tokens, splits them into 500 token chunks
    thus creating observations. This is for fine-tuning with simpletransformers
    later on.
    """
    random.seed(1337)
    _round = 0

    # evaluate size of list of dicts enumerating words and their labels
    size = len(all_records) // num_splits
    print(f"\t\t- Total words in {train_or_test} set: {size}")

    # segment data into `num_splits` chunks
    while _round < num_splits:
        # locate the `_round`th chunk of dicts
        records = all_records[size * _round: size * (_round + 1)]

        # break main chunk of dicts (at this loop round) into smaller chunks of 500 words (`splits` = start/end indices of small chunks)
        splits = create_tokenized_obs(records)
        records = pd.DataFrame(records)

        # cycle through the start/end chunk index tuples
        observations = np.empty(shape=(len(splits), 500, 3), dtype=object)

        with tqdm(range(len(splits))) as S:
            for j in S:
                a, b = splits[j][0], splits[j][1]
                S.set_description(f"                - Splitting data chunk {_round + 1}")

                data_slice = records.iloc[a: b, ].values.tolist()  # collect the 500 word-label dicts between the specified indices
                data_slice = np.pad(data_slice, [(0, 500 - len(data_slice)), (0, 0)], 'empty')

                observations[j] = data_slice  # add each list of 500 dicts to the dataset
                del data_slice

        # shuffle dataset of 500 word-label dicts and save to a txt file
        _round += 1
        random.shuffle(observations)

        out = f'{file_out_nm}_{_round}.npy'
        out_path = os.path.join(file_out_path, out)

        with open(out_path, 'wb') as f:
            np.save(f, observations, allow_pickle=True)

        print(f"\t\t- Output data to file: {out_path}")

        del records
        del observations


def create_tokenized_obs(input_list, num_toks=500, offset=250):
    """
    Given a large set of tokens, determines splits of
    500 token sized observations, with an offset(sliding window) of 250 tokens.
    It is important that first token is capitalized and we fed as many tokens as possible.
    In a real use-case we will not know where splits are so we'll just feed all tokens till limit.
    """
    start = -1
    loop_end = -1
    appends = []

    # cycle through each list of dicts (global index, {word, label}) to break into chunks with a start index `start` and end index `end`
    for ix, i in enumerate(input_list):
        # skip over the first 250 words that have been added to a chunk already (progress to end of offset)
        if ix == loop_end:
            start = -1

        # check if the current word is uppercase
        if i['labels'][-1] == "U" and start == -1:
            start = ix  # start chunk from this uppercase word (i.e. start of a sentence)
            end = ix + num_toks  # the end of that chunk is 500 words later
            appends.append((start, end))  # enumerate the start and end of this chunk
            loop_end = start + offset  # identify end of offset s.t. loop can progress to the end of this offset and start searching again

    # return list of tuples enumerating the start and end index of each chunk of words
    return appends


if __name__ == "__main__":
    # specify dataset to curate
    data = 'news'

    # run data preparation pipeline
    e2e_data(data)
