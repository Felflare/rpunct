# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import re
import json
import random
import pandas as pd


def create_train_datasets():
    output_file_names = []
    download_df()
    for i in ['yelp_polarity_reviews_train.csv', 'yelp_polarity_reviews_test.csv']:
        name = i.split(".")[0]
        split_nm = name.split("_")[-1]
        df_name = name.split("_")[0]
        create_rpunct_dataset(i, f"{name}_data.json")
        output_file_names.append(f"{df_name}_{split_nm}.txt")
        create_training_samples(f"{name}_data.json", f"{df_name}_{split_nm}.txt")
        return output_file_names


def download_df(dir_path=''):
    import tensorflow_datasets as tfds
    data_type = ['train', 'test']
    ds = tfds.load('yelp_polarity_reviews', split=data_type, shuffle_files=True)
    for i in ds:
        i = tfds.as_dataframe(i)
        csv_path = os.path.join(dir_path, f'yelp_polarity_reviews_{i}.csv')
        i.to_csv(csv_path, index=False)


def create_record(row):
    """
    Create labels for Punctuation Restoration task for each token.
    """
    pattern = re.compile("[\W_]+")
    new_obs = []

    observation = eval(row).decode().replace('\\n', ' ').split()

    for obs in observation:
        text_obs = obs.lower()
        text_obs = pattern.sub('', text_obs)

        if not text_obs:
            continue
        if not obs[-1].isalnum():
            new_lab = obs[-1]
        else:
            new_lab = "O"
        if obs[0].isupper():
            new_lab += "U"
        else:
            new_lab += "O"

        new_obs.append({'sentence_id': 0, 'words': text_obs, 'labels': new_lab})
    return new_obs


def create_rpunct_dataset(orig_yelp_dataframe, rpunct_dataset_path='rpunct_data.json'):
    df = pd.read_csv(orig_yelp_dataframe)
    # Filter to only positive examples
    df = df[df['label'] == 1].reset_index(drop=True)
    # Dataframe Shape
    print(f"Dataframe samples: {df.shape}")

    all_records = []
    for i in range(df.shape[0]):
        orig_row = df['text'][i]
        records = create_record(orig_row)
        all_records.extend(records)

    with open(rpunct_dataset_path, 'w') as fp:
        json.dump(all_records, fp)


def create_training_samples(json_loc_file, file_out_nm='train_data', num_splits=5):
    """
    Given a looong list of tokens, splits them into 500 token chunks
    thus creating observations. This is for fine-tuning with simpletransformers
    later on.
    """
    random.seed(1337)
    observations = []
    _round = 0

    while _round < num_splits:
        with open(json_loc_file, 'r') as fp:
            all_records = json.load(fp)

        size = len(all_records) // num_splits
        all_records = all_records[size * _round:size * (_round + 1)]
        splits = create_tokenized_obs(all_records)
        full_data = pd.DataFrame(all_records)
        del all_records

        for i in splits:
            data_slice = full_data.iloc[i[0]:i[1], ]
            observations.append(data_slice.values.tolist())
        _round += 1
        random.shuffle(observations)

        with open(f'{file_out_nm}_{_round}.txt', 'w') as fp2:
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
    for ix, i in enumerate(input_list):
        if ix == loop_end:
            start = -1
        if i['labels'][-1] == "U" and start == -1:
            start = ix
            end = ix + num_toks
            appends.append((start, end))
            loop_end = start + offset

    return appends

if __name__ == "__main__":
    output_file_names = create_train_datasets()
    print(f"Created following files: {output_file_names}")
