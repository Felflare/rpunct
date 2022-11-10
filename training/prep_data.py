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
WORDS_PER_FILE = 15000000


def e2e_data(data_type='news', start_year='2014', end_year='2022', summaries=False, tt_split='90:10', composite_datasets_list=None, composite_data_distinctness=False):
    """
    Full pipeline for compiling and formatting training data from BBC News articles or Yelp reviews
    """
    # generate/collect raw data
    if data_type == 'news':
        # collect data from each JSONL file enumerating all BBC News articles for each year 2014-2022
        print(f"\n> Preparing data from source: BBC News articles")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100
        dataset_path = collate_news_articles(int(start_year), int(end_year), summaries, train_split=split)

    elif data_type == 'reviews':
        # save training/testing datasets from tensorflow to local csv files
        print("\n> Preparing data from source: Yelp reviews")
        dataset_path = download_reviews()

    elif data_type == 'news-transcripts':
        # extract and process transcripts from JSON files
        print(f"\n> Preparing data from source: BBC News transcripts")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100

        data_type = 'transcripts'
        dataset_path = collate_news_transcripts(train_split=split)

    elif data_type == 'composite-news':
        # error checking
        if composite_data_distinctness and len(composite_datasets_list) != 2:
            raise ValueError("If building distinct composite datasets, you must specify exactly TWO included datasets (one for pre-training, one for fine-tuning).")

        # create composte dataset of BBC News articles and transcripts
        print(f"\n> Preparing data from source: BBC News articles & transcripts")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100

        data_type = 'composite'
        dataset_path = create_composite_dataset(distinct=composite_data_distinctness, train_split=split, dataset_names=composite_datasets_list)

    else:
        raise ValueError("Unrecognised data source!")

    total_words = 0
    for key in ['train', 'test']:
        print(f"\n> Generating dataset: {key.upper()}")

        # constuct df of text and labels (punctuation tag per word)
        rpunct_dataset = create_rpunct_dataset(dataset_path, data_type, key, composite_and_distinct=composite_data_distinctness, dataset_names=composite_datasets_list)

        # split data into chunks for model
        primary_rpunct_dataset = rpunct_dataset['A']
        secondary_rpunct_dataset = rpunct_dataset['B']
        del rpunct_dataset

        output_file = f"{data_type}_{key}"
        total_words += create_training_samples(primary_rpunct_dataset, output_file, file_out_path=dataset_path, train_or_test=key)

        # if composite dataset of distinct parts, split/format the fine-tuning half of the data
        if secondary_rpunct_dataset != []:
            output_file = f"{data_type}_{key}_finetuning"
            total_words += create_training_samples(secondary_rpunct_dataset, output_file, file_out_path=dataset_path, train_or_test=key)

    print("\n> Data generation complete")
    print(f"\t* Total no. words in both datasets: {total_words}", end='\n\n')


def check_data_exists(data_type='news', train_or_test='train', start_date='2014', end_date='2022', summaries=False):
    # check whether the training data has been created or not yet
    if data_type == 'news':
        if summaries:
            data_dir = os.path.join(PATH, f'news-summaries')
        else:
            data_dir = os.path.join(PATH, f'news-{start_date}-{end_date}')
    else:
        data_dir = os.path.join(PATH, data_type)

    if data_type == 'news-transcripts':
        data_type = 'transcripts'
    elif data_type == 'composite-news':
        data_type = 'composite'

    data_file_pattern = f'{data_type}_{train_or_test}_*.npy'
    dataset_paths = list(pathlib.Path(data_dir).glob(data_file_pattern))
    data_files_exist = len(dataset_paths) > 0
    print(f"\n> Required data files found at '{data_dir}/{data_file_pattern}'? : {data_files_exist}")

    return data_files_exist


def create_composite_dataset(distinct, train_split, dataset_names):
    # create a collection of all individual datasets needed to construct composite datasets
    all_datasets = []

    for name in dataset_names:
        # collect dataset from file
        if name == 'news-articles':
            # collect articles part of composite dataset (from JSONL files)
            dataset_dir = collate_news_articles(2022, 2022, summaries=False, train_split=1.0, composite=True)
            dataset_path = os.path.join(dataset_dir, 'train_news.csv')
        elif name == 'news-transcripts':
            # collect transcripts part of dataset
            dataset_dir = collate_news_transcripts(train_split=train_split, composite=True)
            dataset_path = os.path.join(dataset_dir, 'train_transcripts.csv')

            # only construct testing data from transcripts dataset
            test_dataset_path_input = os.path.join(dataset_dir, 'test_transcripts.csv')
            transcripts_test = pd.read_csv(test_dataset_path_input)
            transcripts_test.dropna(inplace=True)
            transcripts_test.reset_index(drop=True, inplace=True)

            test_dataset_path_output = os.path.join(dataset_dir, 'test_composite.csv')
            transcripts_test.to_csv(test_dataset_path_output, index=False)
            del transcripts_test
        else:
            raise ValueError("Composite dataset cannot be built with unknown source datasets.")

        # format dataset
        dataset = pd.read_csv(dataset_path)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        # if wanting to create distinct composite dataset (i.e. separate pre-train/tune) then label each dataset to be split up later
        if distinct:
            dataset['source'] = name

        # add to dataset collection
        all_datasets.append(dataset.copy())
        del dataset

    # combine two news datasets together
    composite_data = pd.concat(all_datasets, ignore_index=True)
    del all_datasets

    # shuffle samples (not necessary if splitting into distinct composite datasets)
    if not distinct:
        composite_data = composite_data.sample(frac=1).reset_index(drop=True)

    # save composite dataset to csv file
    csv_path_train = os.path.join(dataset_dir, 'train_composite.csv')
    composite_data.to_csv(csv_path_train, index=False)
    del composite_data

    return dataset_dir


def collate_news_articles(start_date, end_date, summaries, train_split=0.9, composite=False):
    if composite:
        summary_or_body = 'body'
        dataset_path = os.path.join(PATH, f'composite-news')
    elif summaries:
        summary_or_body = 'summary'
        dataset_path = os.path.join(PATH, f'news-summaries')
    else:
        summary_or_body = 'body'
        dataset_path = os.path.join(PATH, f'news-{start_date}-{end_date}')

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
    split = math.ceil(train_split * len(articles))

    train = articles[:split]
    test = articles[split:]

    print(f"\t* Articles in total    : {len(articles)}")
    print(f"\t* Articles in train set: {len(train)}")
    print(f"\t* Articles in test set : {len(test)}")
    del articles

    # save train/test data to csv
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n',' ')
    csv_path_train = os.path.join(dataset_path, 'train_news.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(dataset_path, 'test_news.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    return dataset_path


def collate_news_transcripts(train_split=0.9, composite=False):
    # input transcripts from json files
    print(f"\n> Assembling news transcripts:")
    news_datasets = ['transcripts_2014-17.json', 'transcripts_2020.json']
    articles = np.empty(shape=(0), dtype=object)

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'news_transcripts_2014-20/', dataset_json)

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
    split = math.ceil(train_split * len(articles))
    train = articles[:split]
    test = articles[split:]
    print(f"\t* {train_split:.1f} : {1-train_split:.1f} data split")
    print(f"\t* Speaker segments in total    : {len(articles)}")
    print(f"\t* Speaker segments in train set: {len(train)}")
    print(f"\t* Speaker segments in test set : {len(test)}")
    del articles

    # save train/test data to csv
    if composite:
        dataset_path = os.path.join(PATH, f'composite-news')
    else:
        dataset_path = os.path.join(PATH, f'news-transcripts')

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


def create_rpunct_dataset(path, data_type, split, composite_and_distinct=False, dataset_names=None):
    # load in train/test data
    data_split_path = os.path.join(path, f'{split}_{data_type}.csv')
    data_split = pd.read_csv(data_split_path)
    data_split.dropna(inplace=True)
    data_split.reset_index(drop=True, inplace=True)

    # if we are dealing with a composite dataset of distinct parts, split the prep of each to be processed separately
    if composite_and_distinct and split == 'train':
        # segment distinct articles and transcripts datasets (from composite dataset)
        datasets = []
        for name in dataset_names:
            datasets.append(
                data_split[data_split['source'] == name]['text']
            )

    else:
        datasets = [
            data_split['text']
        ]

    # create records for each dataset part of the composite (if integrated, 'A' == all data; if distinct, 'A' == pre-training and 'B' == fine-tuning)
    all_records = {
        'A': [],
        'B': []
    }

    # constuct df of text and labels (punctuation tag per word) for primary (and possibly secondary) dataset
    for d in range(len(datasets)):
        with tqdm(datasets[d]) as T:
            for article in T:
                T.set_description(f"        * Labelling {split}ing instances ({d})")
                record = create_record(article)  # create a list enumerating each word in a single article and its label: [...{id, word, label}...]
                record_index = list(all_records.keys())[d]
                all_records[record_index].extend(record)
                del record

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


def create_training_samples(words_and_labels, file_out_nm, file_out_path=PATH, train_or_test='train', size=WORDS_PER_FILE):
    """
    Given a looong list of tokens, splits them into 500 token chunks
    thus creating observations. This is for fine-tuning with simpletransformers
    later on.
    """
    random.seed(1337)
    _round = 0
    num_words = len(words_and_labels)

    # determine number of output files dependent on size of dataset
    num_splits = math.ceil(num_words / size)
    print("\t* Formatting data files:")
    print(f"\t\t- No. words in {train_or_test} set: {num_words}")

    # segment primary dataset into `num_splits` chunks
    while _round < num_splits:
        # locate the `_round`th chunk of dicts
        records = words_and_labels[size * _round: size * (_round + 1)]

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

        # shuffle dataset of 500 word-label dicts
        _round += 1
        random.shuffle(observations)

        # save split of dataset to a txt file
        out = f'{file_out_nm}_{_round}.npy'
        out_path = os.path.join(file_out_path, out)

        with open(out_path, 'wb') as f:
            np.save(f, observations, allow_pickle=True)

        print(f"\t\t- Output data to file: {out_path}")

        del records
        del observations

    return num_words


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
