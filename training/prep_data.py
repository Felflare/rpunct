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
WORDS_PER_FILE = 35000000


def e2e_data(data_type='news-articles', start_year='2014', end_year='2022', summaries=False, tt_split='90:10', composite_datasets_list=None, dataset_balance='o'):
    """
    Full pipeline for gathering/loading data and formatting it into training and testing datasets to be used by RPunct.

    Args:
        - data_type (str): a name tag specifying source of the data to be included in the datasets.
        - start_year (str): start year of date range to gather BBC News articles from (requires `data_type = news-articles`).
        - end_year (str): end year of date range to gather BBC News articles from (requires `data_type = news-articles`).
        - summaries (str): toggle between extracting article summaries and article bodies (requires `data_type = news-articles`).
        - tt_split (str): the proportion of collected data to use for training the RPunct model (ratio of percentages in the form `TRAIN:TEST`).
        - composite_datasets_list (list): collection of name tags of multiple data sources to include when constructing a composite dataset (requires `data_type = composite`).
        - dataset_balance (str): specifier to clip dataset sizes to be included in composite dataset in a given ratio to each other (in form `X:Y:...:Z`) (requires `data_type = composite`).
    """
    # generate/collect raw data
    if data_type == 'news-articles':
        # collect data from each JSONL file enumerating all BBC News articles for each year 2014-2022
        print(f"\n> Preparing data from source: BBC News articles")
        data_type = 'news'
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100

        if summaries:
            dataset_path = os.path.join(PATH, f'news-summaries')
            article_size = 'summary'
        else:
            dataset_path = os.path.join(PATH, f'news-{start_year}-{end_year}')
            article_size = 'body'

        collate_news_articles(int(start_year), int(end_year), summary_or_body=article_size, train_split=split, output_directory=dataset_path)

    elif data_type == 'reviews':
        # save training/testing datasets from tensorflow to local csv files
        print("\n> Preparing data from source: Yelp reviews")
        dataset_path = os.path.join(PATH, 'reviews')
        download_reviews(output_directory=dataset_path)

    elif data_type == 'news-transcripts':
        # extract and process transcripts from JSON files
        print(f"\n> Preparing data from source: BBC News transcripts")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100
        data_type = 'transcripts'

        dataset_path = os.path.join(PATH, f'news-transcripts')
        collate_news_transcripts(train_split=split, output_directory=dataset_path)

    elif data_type == 'subtitles':
        # collect data from subtitles JSON files
        print(f"\n> Preparing data from source: subtitles (all genres)")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100

        dataset_path = os.path.join(PATH, f'subtitles')
        collate_subtitles(train_split=split, output_directory=dataset_path)

    elif data_type == 'composite':
        # create composte dataset from multiple sources
        print(f"\n> Preparing data from sources: {composite_datasets_list}")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100

        dataset_path = os.path.join(PATH, 'composite')
        create_composite_dataset(dataset_names=composite_datasets_list, train_split=split, balance=dataset_balance, output_directory=dataset_path)

    else:
        raise ValueError("Unrecognised data source!")

    for split in ['train', 'test']:
        print(f"\n> Generating dataset: {split.upper()}")

        # constuct df of text and labels (punctuation tag per word)
        rpunct_dataset = create_rpunct_dataset(dataset_path, data_type, train_or_test=split)

        create_training_samples(rpunct_dataset, data_type, file_out_path=dataset_path, train_or_test=split)
        del rpunct_dataset

    # remove temporary dataset files
    remove_temp_files(dataset_path, extensions=['csv'], traintest='train')

    print("\n> Data generation complete")


def check_data_exists(data_type='news', train_or_test='train', start_date='2014', end_date='2022', summaries=False):
    """
    Error checking function to ensure training/testing dataset has been constructed before model training/testing commences.

    Args:
        - data_type (str): a name tag specifying source of the data that should be in the training dataset.
        - train_or_test (str): specifier of whether to check for training or testing dataset.
        - start_year (str): start year of date range of BBC News articles training data (requires `data_type = news-articles`).
        - end_year (str): end year of date range of BBC News articles training data (requires `data_type = news-articles`).
        - summaries (bool): specifier for training dataset of article summaries (requires `data_type = news-articles`).
    """
    # check whether the training data has been created or not yet
    if data_type == 'news-articles':
        if summaries:
            data_dir = os.path.join(PATH, f'news-summaries')
        else:
            data_dir = os.path.join(PATH, f'news-{start_date}-{end_date}')
    else:
        data_dir = os.path.join(PATH, data_type)

        if data_type == 'news-transcripts':
            data_type = 'transcripts'

    data_file_pattern = f'{data_type}_{train_or_test}_*.npy'
    dataset_paths = list(pathlib.Path(data_dir).glob(data_file_pattern))
    data_files_exist = len(dataset_paths) > 0 or os.path.isfile(data_dir + "/rpunct_train_set.txt")
    print(f"\n> Required data files found in directory '{data_dir}'? : {data_files_exist}")

    return data_files_exist


def create_rpunct_dataset(directory, data_type, train_or_test='train', make_mc_database=False):
    """
    Converts CSV files of punctuated text data into plaintext labelled word-level training/testing data (list of 'records')
    for supervised learning of punctuation restoration.

    Args:
        - directory (Path): location of directory holding collected data files (train/test CSVs).
        - data_type (str): a name tag specifying source type of the data to be included in the dataset.
        - train_or_test (str): specifier of whether to construct a training or testing dataset.
        - make_mc_database (bool): toggle to use newly labelled data to regrenerate the mixed-case database of plaintext-cased pairs.

    Returns:
        - all_records (lst): a 1D list of 'records' enumerating each word in the training/testing dataset and its punctuation label.
    """
    # load in train/test data
    data_split_path = os.path.join(directory, f'{train_or_test}_{data_type}.csv')
    data_split = pd.read_csv(data_split_path)
    data_split.dropna(inplace=True)
    data_split.reset_index(drop=True, inplace=True)

    # change inter-word dashes to commas
    data_split['text'] = data_split['text'].str.replace(" -", ",")

    # insert a space after intra-word hyphens
    data_split['text'] = data_split['text'].str.replace("-", "- ")

    # only allow creation of mixed-case database if constructing a training dataset
    make_mc_database = make_mc_database and train_or_test == 'train'
    mixed_case_database = {}

    # constuct list of text and labels (punctuation tag per word) for primary (and possibly secondary) dataset
    all_records = []
    with tqdm(data_split['text']) as T:
        T.set_description(f"        * Labelling {train_or_test}ing instances")
        for segment in T:
            # create a list enumerating each word in a single segment/article and its label: [...{id, word, label}...]
            record, mixed_case_instances = create_record(segment, mixed_casing=make_mc_database)

            if make_mc_database:
                mixed_case_database.update(mixed_case_instances)

            all_records.extend(record)
            del record
            del mixed_case_instances

    if make_mc_database:
        with open('rpunct/mixed-casing.json', 'w') as f:
            json.dump(mixed_case_database, f)

    return all_records


def create_record(text, mixed_casing=False):
    """
    Remove punctuation and create data labels for Punctuation Restoration task for each word in a given speaker segment / paragraph of text.

    Args:
        - text (str): a single segment of text from the input CSV data file (the text we want to label).
        - mixed_casing (bool): toggle to add words labelled mixed-case to a new mixed-case database of plaintext-cased pairs.

    Returns:
        - new_obs (lst): a collection of dictionaries specifying each word-label pair to be added to the dataset (dict holds the index, plaintext, and punctuation label)
        - mixed_case (dict): output collection of plaintext-cased pairs relating to identified mixed-case words to be added to the mixed-case database (empty if `mixed_casing = False`)
    """
    pattern = re.compile(r"[^0-9a-zA-Z']")
    new_obs = []
    mixed_case = {}

    # convert string of text (from segment) into a list of words
    observation = text.replace('\\n', ' ').split()

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
            if mixed_casing:
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


def create_training_samples(words_and_labels, data_type, file_out_path=PATH, train_or_test='train', size=WORDS_PER_FILE):
    """
    Splits up a list of words and their labels (that will form the training/testing dataset) into chunks of 500 instances (observations)
    that can be input into the BERT-based transformer model.
    Then output this segmented dataset of observations to multiple NPY files of a fixed size each holding part of the training/testing dataset.

    Args:
        - words_and_labels (lst): a 1D list of 'records' enumerating each word in the training/testing dataset and its punctuation label.
        - data_type (str): a name tag specifying source type of the data to be included in the dataset.
        - file_out_path (Path): location of directory in which to store the output NPY data files.
        - train_or_test (str): specifier of whether to sace data as a training or testing dataset.
        - size (int): number of observations to save to a single output NPY file.
    """
    # determine number of output files dependent on size of dataset
    num_words = len(words_and_labels)
    num_splits = math.ceil(num_words / size)
    random.seed(42)
    _round = 0

    print("\t* Formatting data files:")
    print(f"\t\t- No. words in {train_or_test} set  : {num_words}")

    # segment primary dataset into `num_splits` chunks
    while _round < num_splits:
        # locate the `_round`th chunk of dicts
        records = words_and_labels[size * _round: size * (_round + 1)]

        # break main chunk of dicts (at this loop round) into smaller chunks of 500 words (`splits` = start/end indices of observation chunks)
        splits = create_tokenized_obs(records)
        records = pd.DataFrame(records)

        # cycle through the start/end chunk index tuples
        observations = np.empty(shape=(len(splits), 500, 3), dtype=object)

        with tqdm(range(len(splits))) as S:
            S.set_description(f"                - Splitting data chunk {_round + 1} ")
            for j in S:
                a, b = splits[j][0], splits[j][1]
                data_slice = records.iloc[a: b, ].values.tolist()  # collect the 500 word-label dicts between the specified indices
                data_slice = np.pad(data_slice, [(0, 500 - len(data_slice)), (0, 0)], 'empty')

                observations[j] = data_slice  # add each list of 500 dicts to the dataset
                del data_slice

        # shuffle dataset of 500 word-label dicts
        _round += 1
        random.shuffle(observations)

        # save split of dataset to a txt file
        out_file = f"{data_type}_{train_or_test}_{_round}.npy"
        out_path = os.path.join(file_out_path, out_file)

        with open(out_path, 'wb') as f:
            np.save(f, observations, allow_pickle=True)

        print(f"\t\t- Output data to file               : {out_path}")
        del records
        del observations


def create_tokenized_obs(input_list, num_toks=500, offset=250):
    """
    Generate the start/end indices of the observations (of `num_toks` instances) segmented from a given set of word-label instances,
    with an offset (sliding window) of `offset` instances.
    It is important that first token is capitalized and we use as many tokens as possible.

    Args:
        - input_list (lst): collection of word-label pairs to segment into observations.
        - num_toks (int): number of instances to include in each observation (chunk).
        - offset (int): offset from the begining of the previous chunk to start collecting data for the next chunk (gives an overlap of `num_toks -  offset` instances).

    Returns:
        - indices (lst): collection of index pairs specifying the start and end index of each observation segmented from the input data.
    """
    start = -1
    loop_end = -1
    indices = []

    # cycle through each list of dicts (global index, {word, label}) to break into chunks with a start index `start` and end index `end`
    for ix, i in enumerate(input_list):
        # skip over the first 250 words that have been added to a chunk already (progress to end of offset)
        if ix == loop_end:
            start = -1

        # check if the current word is uppercase (so we can start all observations at the beginning of a sentence)
        if i['labels'][-1] == "C" and start == -1:
            start = ix  # start chunk from this uppercase word (i.e. start of a sentence)
            end = ix + num_toks  # the end of that chunk is 500 words later
            indices.append((start, end))  # enumerate the start and end of this chunk
            loop_end = start + offset  # identify end of offset s.t. loop can progress to the end of this offset and start searching again

    # return list of tuples enumerating the start and end index of each chunk of words
    return indices


if __name__ == "__main__":
    # specify dataset to curate
    data = 'news-articles'

    # run data preparation pipeline
    e2e_data(data)
