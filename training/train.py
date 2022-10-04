# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import json
import os
import pandas as pd
from simpletransformers.ner import NERModel

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
TRAIN_DATASETS = ['yelp_train_1.txt', 'yelp_train_2.txt', 'yelp_train_3.txt', 'yelp_train_4.txt']
PATH = './training/datasets/'


def e2e_train(use_cuda=True, validation=True, print_stats=False):
    # generate correctly formatted training data
    print("\nPreparing training data")
    prepare_data(TRAIN_DATASETS, validation=validation, print_stats=print_stats)

    # create a simpletransformer model and use data to train it
    print("\nBuilding & training model")
    model, steps, tr_details = train_model(use_cuda=use_cuda, validation=validation)
    print(f"Steps: {steps}; Train details: {tr_details}")

    return model


def prepare_data(datasets, output_txt='rpunct_train_set.txt', validation=True, print_stats=False):
    """
    Prepares data from Original text into Connnl formatted datasets ready for training
    In addition constraints label space to only labels we care about
    """
    # load formatted data generated through `prep_data.py`
    token_data = load_datasets(datasets)

    # remove any invalid labels
    clean_up_labels(token_data, VALID_LABELS)

    # split train/test datasets, and convert each to a text file
    if validation:
        # training set
        train_set = token_data[:int(len(token_data) * 0.90)]
        train_set_path = os.path.join(PATH, output_txt)
        create_text_file(train_set, train_set_path)
        print(f"\tTraining dataset shape: ({len(train_set)}, {len(train_set[0])}, {len(train_set[0][0])})")

        # validation set
        val_set = token_data[-int(len(token_data) * 0.10):]
        val_set_path = os.path.join(PATH, 'rpunct_val_set.txt')
        create_text_file(val_set, val_set_path)
        print(f"\tValidation dataset shape: ({len(val_set)}, {len(val_set[0])}, {len(val_set[0][0])})")
    else:
        # if not having a validation set, just create one dataset
        dataset = token_data.copy()
        dataset_path = os.path.join(PATH, output_txt)
        create_text_file(dataset, dataset_path)
        print(f"\tDataset shape: ({len(dataset)}, {len(dataset[0])}, {len(dataset[0][0])})")

    # output statistics of each dataset
    if print_stats:
        train_stats = get_label_stats(train_set)
        train_stats = pd.DataFrame.from_dict(train_stats, orient='index', columns=['count'])
        val_stats = get_label_stats(val_set)
        val_stats = pd.DataFrame.from_dict(val_stats, orient='index', columns=['count'])

        print(f"\tTraining data statistics:")
        print(train_stats)

        print(f"\n\tValidation data statistics:")
        print(val_stats)


def load_datasets(dataset_paths):
    """
    Given a list of data paths returns a single data object containing all data slices
    """
    token_data = []
    for d_set in dataset_paths:
        dataset_path = os.path.join(PATH, d_set)
        with open(dataset_path, 'r') as fp:
            data_slice = json.load(fp)

        token_data.extend(data_slice)
        del data_slice

    return token_data


def clean_up_labels(dataset, valid_labels):
    """
    Given a list of Valid labels cleans up the dataset
    by limiting to only the labels available.

    In addition prepares observations for training.
    """
    for ix, i in enumerate(dataset):
        for tok in i:
            tok[0] = ix
            if tok[2] not in valid_labels:
                case = tok[2][-1]
                tok[2] = f"O{case}"
                if len(tok[2]) < 2:
                    tok[2] = "OO"


def create_text_file(dataset, name):
    """
    Create Connl ner format file
    """
    with open(name, 'w') as fp:
        for obs in dataset:
            for tok in obs:
                line = tok[1] + " " + tok[2] + '\n'
                fp.write(line)
            fp.write('\n')


def get_label_stats(dataset):
    """
    Generates frequency of different labels in the dataset.
    """
    calcs = {}
    for i in dataset:
        for tok in i:
            if tok[2] not in calcs.keys():
                calcs[tok[2]] = 1
            else:
                calcs[tok[2]] += 1

    return calcs


def train_model(train_data_txt='rpunct_train_set.txt', val_data_txt='rpunct_val_set.txt', use_cuda=True, validation=True):
    """
    Trains simpletransformers model
    """
    # Create a NERModel
    print("\tBuilding NER model", end='\n\n')
    model = NERModel(
        "bert",
        "bert-base-uncased",
        labels=VALID_LABELS,
        use_cuda=use_cuda,
        args={
            "evaluate_during_training": validation,
            "overwrite_output_dir": True,
            "num_train_epochs": 3,
            "max_seq_length": 512,
            "lazy_loading": True
        }
    )

    # Train the model
    train_data_path = os.path.join(PATH, train_data_txt)
    val_data_path = os.path.join(PATH, val_data_txt)
    print(f"\n\tTraining model on dataset: {train_data_path}")
    print(f"\tValidate model during training: {validation}", end='\n\n')

    steps, tr_details = model.train_model(train_data_path, eval_data=val_data_path)

    return model, steps, tr_details


if __name__ == "__main__":
    e2e_train()
