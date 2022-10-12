# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import re
import sys
import json
import math
import random
import pathlib
import pandas as pd
import tensorflow_datasets as tfds

REVIEWS_DATASETS = ['yelp_polarity_reviews_train.csv', 'yelp_polarity_reviews_test.csv']
NEWS_DATASETS = ['news_2022.jsonl', 'news_2021.jsonl', 'news_2020.jsonl', 'news_2019.jsonl', 'news_2018.jsonl', 'news_2017.jsonl', 'news_2016.jsonl', 'news_2015.jsonl', 'news_2014.jsonl']
PATH = './training/datasets/'
NEWS_PATH = './training/datasets/news_data/'
SUMMARY_OR_BODY = 'summary'


def e2e_data(data_type='news'):
    if data_type == 'news':
        print(f"\nPreparing data from source: BBC News")
        news_data_pipeline()  # construct training and testing data files from BBC News articles
    elif data_type == 'reviews':
        print(f"\nPreparing data from source: Yelp reviews")
        yelp_data_pipeline()  # construct training and testing data files from Yelp reviews
    else:
        raise ValueError('Unknown data source')


def news_data_pipeline():
    """
    Full pipeline for compiling and formatting training data from BBC News articles
    """
    # collect data from each JSONL file enumerating all BBC News articles for each year 2014-2022
    all_news_data = collate_news_articles()

    for key in all_news_data.keys():
        print(f"\nGenerating dataset: {key.upper()}")

        # constuct df of text and labels (punctuation tag per word)
        print("\nLabelling data instances")
        data_split = all_news_data[key]
        rpunct_dataset_file = f"news_{key}_data.json"
        create_rpunct_dataset(data_split, rpunct_dataset_file, data_type='news')

        # split data into chunks for model
        print("\nGenerating data samples")
        split_dataset_file = f"news_{key}"
        create_training_samples(rpunct_dataset_file, split_dataset_file)


def yelp_data_pipeline():
    """
    Full pipeline for downloading and formatting training Yelp reviews data
    """
    # save training/testing datasets from tensorflow to local csv files
    print("\nDownloading csv files")
    download_reviews()

    # formatting training and testing datasets correctly to be input into rpunct
    for dataset_txt in REVIEWS_DATASETS:
        # collect metadata from csv files
        name = dataset_txt.split(".")[0]  # remove extension
        split_nm = name.split("_")[-1]  # collect train/test label
        df_name = name.split("_")[0]  # collect dataset name
        print(f"\nGenerating dataset: {split_nm.upper()}")

        # format data as text and punctuation tag labels
        print("\nLabelling data instances")
        rpunct_dataset_file = f"{name}_data.json"
        create_rpunct_dataset(dataset_txt, rpunct_dataset_file, data_type='reviews')

        print("\nGenerating data samples")
        split_dataset_file = f"{df_name}_{split_nm}"
        create_training_samples(rpunct_dataset_file, split_dataset_file)


def check_data_exists(data_type='news', train_or_test='train'):
    # check whether the training data has been created or not yet
    if data_type == 'news':
        data_file_pattern = f'news_{train_or_test}_*.txt'
        dataset_paths = list(pathlib.Path(PATH).glob(data_file_pattern))

    else:  # data_type == 'reviews'
        data_file_pattern = f'yelp_{train_or_test}_*.txt'
        dataset_paths = list(pathlib.Path(PATH).glob(data_file_pattern))

    if len(dataset_paths) == 0:
        print(f"Dataset files found: False")
        return False

    data_file_pattern = os.path.join(PATH, f'rpunct_{train_or_test}_set.txt')
    final_data_file = os.path.isfile(data_file_pattern)
    print(f"\nDataset files found: {final_data_file}")

    return final_data_file


def collate_news_articles():
    summaries = []

    for dataset_json in NEWS_DATASETS:
        json_path = os.path.join(NEWS_PATH, dataset_json)

        with open(json_path, 'r') as fp:
            for line in fp:
                summaries.append(json.loads(line)[SUMMARY_OR_BODY])

    print(f"\nAssembled {len(summaries)} news article summaries.")

    # split dataset into training and testing instances
    random.shuffle(summaries)
    split = math.ceil(0.9 * len(summaries))
    train = summaries[:split]
    test = summaries[split:]
    print(f"\tSummaries (lines of text) in train set: {len(train)}")
    print(f"\tSummaries (lines of text) in test set : {len(test)}")

    data = {
        'train': train,
        'test': test
    }

    return data

def download_reviews():
    # get distinct datasets for training and testing from tensorflow_datasets
    data_type = ['train', 'test']
    ds = tfds.load('yelp_polarity_reviews', split=data_type, shuffle_files=True)

    # save these two datasets as csv files
    for ind in range(len(ds)):
        i = tfds.as_dataframe(ds[ind])
        csv_path = os.path.join(PATH, f'yelp_polarity_reviews_{data_type[ind]}.csv')
        i.to_csv(csv_path, index=False)


def create_rpunct_dataset(unformatted_data, output_data_file='rpunct_data.json', data_type='reviews'):
    if data_type == 'reviews':
        # read in the csv file as a dataframe
        unformatted_data_path = os.path.join(PATH, unformatted_data)
        df = pd.read_csv(unformatted_data_path)

        # Filter to only positive examples
        df = df[df['label'] == 1].reset_index(drop=True)
    elif data_type == 'news':
        df = pd.DataFrame(unformatted_data, columns =['text'])
    else:
        raise ValueError('Unknown data source')

    # Dataframe Shape
    print(f"\tDataframe samples: {df.shape}")

    # constuct df of text and labels (punctuation tag per word)
    all_records = []
    for i in range(df.shape[0]):
        orig_row = df['text'][i]  # fetch a single row of text data
        records = create_record(orig_row, data_type=data_type)  # create a list enumerating each word in the row and its label: [...{id, word, label}...]
        all_records.extend(records)

    # save the list of all {word, label} dicts to a json file
    output_data_path = os.path.join(PATH, output_data_file)
    with open(output_data_path, 'w') as fp:
        json.dump(all_records, fp)


def create_record(row, data_type='reviews'):
    """
    Create labels for Punctuation Restoration task for each token.
    """
    pattern = re.compile("[\W_]+")
    new_obs = []

    # convert string of text (from row) into a list of words
    # row: str -> observation: list[str]
    if data_type == 'reviews':
        observation = eval(row).decode().replace('\\n', ' ').split()
    elif data_type == 'news':
        observation = row.replace('\\n', ' ').split()
    else:
        raise ValueError('Unknown data source')


    # remove punctuation of each word, and label it with a tag representing what punctuation it did have
    for obs in observation:
        text_obs = obs.lower()  # set lowercase
        text_obs = pattern.sub('', text_obs)  # check whether word is the empty string

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


def create_training_samples(json_loc_file, file_out_nm='train_data', num_splits=5):
    """
    Given a looong list of tokens, splits them into 500 token chunks
    thus creating observations. This is for fine-tuning with simpletransformers
    later on.
    """
    random.seed(1337)
    _round = 0

    # segment data into `num_splits` chunks
    while _round < num_splits:
        # open json file of list of dicts enumerating words and their labels
        json_path = os.path.join(PATH, json_loc_file)
        with open(json_path, 'r') as fp:
            all_records = json.load(fp)

        # locate the `_round`th chunk of dicts
        size = len(all_records) // num_splits
        all_records = all_records[size * _round:size * (_round + 1)]

        # break main chunk of dicts (at this loop round) into smaller chunks of 500 words (`splits` = start/end indices of small chunks)
        splits = create_tokenized_obs(all_records)
        full_data = pd.DataFrame(all_records)
        del all_records

        # cycle through the start/end chunk index tuples
        observations = []

        for i in splits:
            data_slice = full_data.iloc[i[0]:i[1], ]  # collect the 500 word-label dicts between the specified indices
            observations.append(data_slice.values.tolist())  # add each list of 500 dicts to the dataset

        # shuffle dataset of 500 word-label dicts and save to a txt file
        _round += 1
        random.shuffle(observations)

        out = f'{file_out_nm}_{_round}.txt'
        print(f"\tOutput data to file: {out}")

        out_path = os.path.join(PATH, out)
        with open(out_path, 'w') as fp2:
            json.dump(observations, fp2)

        del full_data
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
