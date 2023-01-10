# -*- coding: utf-8 -*-
# 💾⚙️🔮
__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import re
import math
import json
import random
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from training.get_data import *

PATH = './training/datasets/'
WORDS_PER_FILE = 40000000


def e2e_data(data_type='news', start_year='2014', end_year='2022', summaries=False, tt_split='90:10', composite_datasets_list=None, composite_data_distinctness=False, dataset_balance='o'):
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

    elif data_type == 'subtitles':
        # collect data from subtitles JSON files
        print(f"\n> Preparing data from source: subtitles (all genres)")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100
        dataset_path = collate_subtitles(train_split=split)

    elif data_type == 'composite':
        # error checking
        if composite_data_distinctness and composite_datasets_list is  None:
            raise ValueError("If building distinct composite datasets, you must specify exactly TWO included datasets (one for pre-training, one for fine-tuning).")

        # create composte dataset of BBC News articles and transcripts
        print(f"\n> Preparing data from source: BBC News articles & transcripts")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100

        dataset_path = create_composite_dataset(distinct=composite_data_distinctness, train_split=split, dataset_names=composite_datasets_list, balance=dataset_balance)

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

    # remove temporary dataset files
    remove_temp_files(dataset_path, extensions=['csv'], traintest='train')

    print("\n> Data generation complete")
    print(f"\t* Total no. words in both datasets: {total_words}", end='\n\n')


def check_data_exists(data_type='news', train_or_test='train', start_date='2014', end_date='2022', summaries=False, finetuning=False):
    # check whether the training data has been created or not yet
    if data_type == 'news':
        if summaries:
            data_dir = os.path.join(PATH, f'news-summaries')
        else:
            data_dir = os.path.join(PATH, f'news-{start_date}-{end_date}')
    elif data_type[:15] == 'composite-news-':
        data_dir = os.path.join(PATH, data_type)
        data_type = 'composite'
    else:
        data_dir = os.path.join(PATH, data_type)

        if data_type == 'news-transcripts':
            data_type = 'transcripts'

    if finetuning:
        train_or_test = 'train_finetuning'

    data_file_pattern = f'{data_type}_{train_or_test}_*.npy'
    dataset_paths = list(pathlib.Path(data_dir).glob(data_file_pattern))
    data_files_exist = len(dataset_paths) > 0
    training_txt_exists = os.path.isfile(data_dir + "/rpunct_train_set.txt")
    check = data_files_exist or training_txt_exists
    print(f"\n> Required data files found in directory '{data_dir}'? : {check}")

    return check


def create_rpunct_dataset(path, data_type, split, composite_and_distinct=False, dataset_names=None, make_mc_database=False):
    # load in train/test data
    data_split_path = os.path.join(path, f'{split}_{data_type}.csv')
    data_split = pd.read_csv(data_split_path)
    data_split.dropna(inplace=True)
    data_split.reset_index(drop=True, inplace=True)

    # change inter-word dashes to commas
    data_split['text'] = data_split['text'].str.replace(" -", ",")

    # insert a space after intra-word hyphens
    data_split['text'] = data_split['text'].str.replace("-", "- ")

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

    if make_mc_database:
        mixed_case = {}
    else:
        mixed_case = None

    # constuct df of text and labels (punctuation tag per word) for primary (and possibly secondary) dataset
    for d in range(len(datasets)):
        with tqdm(datasets[d]) as T:
            T.set_description(f"        * Labelling {split}ing instances ({d})")
            for segment in T:
                record, mixed_case = create_record(segment, mixed_case)  # create a list enumerating each word in a single segment/article and its label: [...{id, word, label}...]
                record_index = list(all_records.keys())[d]
                all_records[record_index].extend(record)
                del record

    if split == 'train' and make_mc_database:
        with open('rpunct/mixed-casing.json', 'w') as f:
            json.dump(mixed_case, f)

    return all_records

def create_record(row, mixed_case):
    """
    Create labels for Punctuation Restoration task for each token.
    """
    pattern = re.compile(r"[^0-9a-zA-Z']")
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
        elif text_obs[-1] == "'":  # remove trailing punctuation (only leave mid-word apostrophes)
            text_obs = text_obs[:-1]

        # if there is a punctuation mark after the word, add it to the label
        if not obs[-1].isalnum():
            new_lab = obs[-1]
        else:
            new_lab = "O"  # `O` => no punctuation

        stripped_obs = re.sub(r"[^0-9a-zA-Z]", "", obs)
        if stripped_obs is '':
            continue

        # if the word is lowercase/capitalised/uppercase/mixed-case, add a descriptor to the label
        if stripped_obs.islower() or stripped_obs.isnumeric():
            new_lab += "O"  # `xO` => lowercase (incl. numbers)
        elif stripped_obs.isupper():
            new_lab += "U"  # `xU` => uppercase
        elif stripped_obs[0].isupper() and (len(stripped_obs) == 1 or stripped_obs[1:].islower()):
            new_lab += "C"  # `xC` => capitalised
        else:
            new_lab += "M"  # `xM` => mixed-case

            # populate database of mixed-case instances
            if mixed_case is not None:
                less_stripped_obs = re.sub(r"[^0-9a-zA-Z']", "", obs)

                if not less_stripped_obs[0].isalnum():
                    less_stripped_obs = less_stripped_obs[1:]

                if not less_stripped_obs[-1].isalnum():
                    less_stripped_obs = less_stripped_obs[:-1]

                if less_stripped_obs[-2:] != "'s" and less_stripped_obs[-1:] != 's' and text_obs not in mixed_case.keys():
                    mixed_case.update({text_obs: less_stripped_obs})

        # add the word and its label to the dataset
        new_obs.append({'sentence_id': 0, 'words': text_obs, 'labels': new_lab})

        del text_obs
        del new_lab

    return new_obs, mixed_case


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
            S.set_description(f"                - Splitting data chunk {_round + 1}")
            for j in S:
                a, b = splits[j][0], splits[j][1]
                data_slice = records.iloc[a: b, ].values.tolist()  # collect the 500 word-label dicts between the specified indices
                data_slice = np.pad(data_slice, [(0, 500 - len(data_slice)), (0, 0)], 'empty')

                observations[j] = data_slice  # add each list of 500 dicts to the dataset
                del data_slice

        # shuffle dataset of 500 word-label dicts
        _round += 1
        random.seed(42)
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
        if i['labels'][-1] == "C" and start == -1:
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
