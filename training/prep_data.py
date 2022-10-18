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
import pandas as pd
import tensorflow_datasets as tfds
from tqdm import tqdm

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
PATH = './training/datasets/'
NEWS_PATH = './training/datasets/news_data/'
NO_OUTPUT_FILES = 5
SUMMARY_OR_BODY = 'body'
NEWS_START_YEAR = 2022
NEWS_END_YEAR = 2022


def e2e_data(data_type='news'):
    """
    Full pipeline for compiling and formatting training data from BBC News articles or Yelp reviews
    """
    # generate/collect raw data
    if data_type == 'news':
        # collect data from each JSONL file enumerating all BBC News articles for each year 2014-2022
        print(f"\n> Preparing data from source: BBC News ({SUMMARY_OR_BODY})")
        all_data = collate_news_articles()
    else:  # data_type == 'reviews'
        # save training/testing datasets from tensorflow to local csv files
        print("\n> Preparing data from source: Yelp reviews")
        all_data = download_reviews()

    for key in ['train', 'test']:
        print(f"\n> Generating dataset: {key.upper()}")

        # constuct df of text and labels (punctuation tag per word)
        # print("\t* Labelling data instances", end='\n\n')
        data_split = all_data[key]
        words_and_labels = create_rpunct_dataset(data_split)

        # split data into chunks for model
        print("\t* Generating data samples")
        output_file = f"{data_type}_{key}"
        create_training_samples(words_and_labels, output_file, num_splits=NO_OUTPUT_FILES, train_or_test=key)

    print("\n> Data generation complete", end='\n\n')


def check_data_exists(data_type='news', train_or_test='train'):
    # check whether the training data has been created or not yet
    data_file_pattern = f'{data_type}_{train_or_test}_*.txt'
    dataset_paths = list(pathlib.Path(PATH).glob(data_file_pattern))
    data_files_exist = len(dataset_paths) == NO_OUTPUT_FILES
    print(f"\n> Required data files found: {data_files_exist} ({data_file_pattern})")

    return data_files_exist


def collate_news_articles(start_date=NEWS_START_YEAR, end_date=NEWS_END_YEAR):
    news_datasets = [f'news_{date}.jsonl' for date in range(start_date, end_date + 1)]
    articles = []

    for dataset_json in news_datasets:
        json_path = os.path.join(NEWS_PATH, dataset_json)

        with open(json_path, 'r') as fp:
            for line in fp:
                obj = json.loads(line)
                articles.append(str(obj[SUMMARY_OR_BODY]))

    print(f"\n> Assembling news article {SUMMARY_OR_BODY[:-1]}ies (one line per {SUMMARY_OR_BODY}):")

    # train-test split
    random.shuffle(articles)
    split = math.ceil(0.9 * len(articles))
    train = articles[:split]
    test = articles[split:]
    print(f"\t* Lines in total    : {len(articles)}")
    print(f"\t* Lines in train set: {len(train)}")
    print(f"\t* Lines in test set : {len(test)}")

    # compile into single dataframe
    data = {
        'train': pd.DataFrame(train, columns=['text']),
        'test': pd.DataFrame(test, columns=['text'])
    }

    return data


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
    test = test[test['label'] == 1].reset_index(drop=True)

    # compile into single dataframe
    data = {
            'train': train,
            'test': test
    }

    return data


def create_rpunct_dataset(df):
    # constuct df of text and labels (punctuation tag per word)
    all_records = []
    with tqdm(range(df.shape[0])) as R:
        for i in R:
            R.set_description("        * Labelling data instances")
            orig_row = df['text'][i]  # fetch a single row of text data
            records = create_record(orig_row)  # create a list enumerating each word in the row and its label: [...{id, word, label}...]
            all_records.extend(records)

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

    return new_obs


def create_training_samples(all_records, file_out_nm='train_data', num_splits=5, train_or_test='train'):
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
        all_records = all_records[size * _round: size * (_round + 1)]

        # break main chunk of dicts (at this loop round) into smaller chunks of 500 words (`splits` = start/end indices of small chunks)
        splits = create_tokenized_obs(all_records)
        full_data = pd.DataFrame(all_records)

        # cycle through the start/end chunk index tuples
        observations = []
        for i in splits:
            data_slice = full_data.iloc[i[0]:i[1], ]  # collect the 500 word-label dicts between the specified indices
            observations.append(data_slice.values.tolist())  # add each list of 500 dicts to the dataset

        # shuffle dataset of 500 word-label dicts and save to a txt file
        _round += 1
        random.shuffle(observations)

        out = f'{file_out_nm}_{_round}.txt'
        print(f"\t\t- Output data to file: {out}")

        out_path = os.path.join(PATH, out)
        with open(out_path, 'w') as fp2:
            json.dump(observations, fp2)


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
