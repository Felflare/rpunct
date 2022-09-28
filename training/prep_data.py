# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import re
import json
import random
import pandas as pd
import tensorflow_datasets as tfds

TRAIN_DATASETS = ['yelp_polarity_reviews_train.csv', 'yelp_polarity_reviews_test.csv']
PATH = './training/datasets/'

# full pipeline for downloading and formatting training data
def create_train_datasets():
    output_file_names = []

    # save training/testing datasets from tensorflow to local csv files
    print("\nDownloading csv files")
    download_df()

    # formatting training and testing datasets correctly to be input into rpunct
    for i in TRAIN_DATASETS:
        # collect metadata from csv files
        name = i.split(".")[0]  # remove extension
        split_nm = name.split("_")[-1]  # collect train/test label
        df_name = name.split("_")[0]  # collect dataset name
        print(f"\nGenerating dataset: {split_nm}")

        # format data as text and punctuation tag labels
        print("\nLabelling data instances")
        data_in = os.path.join(PATH, i)
        data_out = os.path.join(PATH, f"{name}_data.json")
        create_rpunct_dataset(data_in, data_out)

        print("\nGenerating training samples")
        output_files = create_training_samples(f"{name}_data.json", f"{df_name}_{split_nm}")
        output_file_names.extend(output_files)

    return output_file_names


def download_df():
    # get distinct datasets for training and testing from tensorflow_datasets
    data_type = ['train', 'test']
    ds = tfds.load('yelp_polarity_reviews', split=data_type, shuffle_files=True)

    # save these two datasets as csv files
    for ind in range(len(ds)):
        i = tfds.as_dataframe(ds[ind])
        csv_path = os.path.join(PATH, f'yelp_polarity_reviews_{data_type[ind]}.csv')
        i.to_csv(csv_path, index=False)


def create_rpunct_dataset(orig_yelp_dataframe, rpunct_dataset_path='rpunct_data.json'):
    # read in the csv file as a dataframe
    df = pd.read_csv(orig_yelp_dataframe)

    # Filter to only positive examples
    df = df[df['label'] == 1].reset_index(drop=True)

    # Dataframe Shape
    print(f"\tDataframe samples: {df.shape}")

    # constuct df of text and labels (punctuation tag per word)
    all_records = []
    for i in range(df.shape[0]):
        orig_row = df['text'][i]  # fetch a single row of text data
        records = create_record(orig_row)  # create a list enumerating each word in the row and its label: [...{id, word, label}...]
        all_records.extend(records)

    # save the list of all {word, label} dicts to a json file
    with open(rpunct_dataset_path, 'w') as fp:
        json.dump(all_records, fp)


def create_record(row):
    """
    Create labels for Punctuation Restoration task for each token.
    """
    pattern = re.compile("[\W_]+")
    new_obs = []

    # convert string of text (from row) into a list of words
    # row: str -> observation: list[str]
    observation = eval(row).decode().replace('\\n', ' ').split()

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
    output_files = []

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
        output_files.append(out)
        print(f"\tOutput data to file: {out}")

        out_path = os.path.join(PATH, out)
        with open(out_path, 'w') as fp2:
            json.dump(observations, fp2)

        del full_data
        del observations

    return output_files


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
        # skip over words if they have been added to a chunk already (progress to end of that chunk)
        if ix == loop_end:
            start = -1

        # check if the current word is uppercase
        if i['labels'][-1] == "U" and start == -1:
            start = ix  # start chunk from this uppercase word (i.e. start of a sentence)
            end = ix + num_toks  # the end of that chunk is 500 words later
            appends.append((start, end))  # enumerate the start and end of this chunk
            loop_end = start + offset  # identify end of chunk s.t. loop can progress to the end of this chunk and start searching again

    # return list of tuples enumerating the start and end index of each chunk of words
    return appends


if __name__ == "__main__":
    output_file_names = create_train_datasets()
    print(f"\nCreated following files: {output_file_names}")
