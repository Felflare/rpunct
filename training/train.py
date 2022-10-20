# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import json
import math
import pathlib
import numpy as np
import pandas as pd
from simpletransformers.ner import NERModel
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(10, 7), 'figure.dpi':100, 'savefig.dpi':100})

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
PATH = './training/datasets/'


def e2e_train(data_type='reviews', use_cuda=True, validation=False, dataset_stats=False, training_plot=False, epochs=3):
    """
    Training pipeline to format training dataset, build model, and train it.
    """
    if data_type not in ['news', 'reviews']:
        raise ValueError('Unknown data source')

    # generate correctly formatted training data
    prepare_data(data_type=data_type, validation=validation, print_stats=dataset_stats)

    # create a simpletransformer model and use data to train it
    print("\n> Building & training model")
    model, steps, tr_details = train_model(use_cuda=use_cuda, validation=validation, epochs=epochs)
    print(f"\n\t* Steps: {steps}; Train details: {tr_details}")

    # plot the progression/convergence over training/validation
    if validation and training_plot:
        plot_training(tr_details)

    print("\n> Model training complete", end='\n\n')

    return model


def prepare_data(data_type='reviews', train_or_test='train', validation=True, print_stats=False):
    """
    Prepares data from Original text into Connnl formatted datasets ready for training
    In addition constraints label space to only labels we care about
    """
    # load formatted data generated through `prep_data.py`
    token_data = load_datasets(data_type, train_or_test)

    # remove any invalid labels
    clean_up_labels(token_data, VALID_LABELS)

    # split train/test datasets, convert each to a text file, and print dataset stats if desired
    if validation:
        # training & validation sets
        split = math.ceil(0.9 * len(token_data))
        train_set = token_data[:split].copy()

        val_set = token_data[split:].copy()
        val_output_txt='rpunct_val_set.txt'
        val_set_path = os.path.join(PATH, val_output_txt)

        # format validaton set as Connl NER txt file
        create_text_file(val_set, val_set_path)
        print(f"\t* Validation dataset shape: ({len(val_set)}, {len(val_set[0])}, {len(val_set[0][0])})")

        # print label distribution in validation set
        if print_stats:
            val_stats = get_label_stats(val_set)
            val_stats = pd.DataFrame.from_dict(val_stats.items())
            val_stats.columns = ['Punct Tag', 'Count']
    else:
        train_set = token_data.copy()

    # format training set as Connl NER txt file
    output_txt = f'rpunct_{train_or_test}_set.txt'
    train_set_path = os.path.join(PATH, output_txt)
    create_text_file(train_set, train_set_path)
    print(f"\t* {train_or_test.capitalize()}ing dataset shape: ({len(train_set)}, {len(train_set[0])}, {len(train_set[0][0])})")

    # print label distribution in training set
    if print_stats:
        train_stats = get_label_stats(train_set)
        train_stats = pd.DataFrame.from_dict(train_stats.items())
        train_stats.columns = ['Punct Tag', 'Count']

        print(f"\t* {train_or_test.capitalize()}ing data statistics:")
        print(train_stats)

        if validation:
            print(f"\t* Validation data statistics:")
            print(val_stats)


def load_datasets(data_type='reviews', train_or_test='train'):
    """
    First, locate the data files generated from running `prep_data.py`.
    Then, given this list of data paths return a single data object containing all data slices.
    """
    # find training data files
    print(f"\n> Loading data from source: {data_type.upper()}")
    data_file_pattern = f'{data_type}_{train_or_test}_*.npy'
    dataset_paths = list(pathlib.Path(PATH).glob(data_file_pattern))

    if len(dataset_paths) == 0:
        raise FileNotFoundError("No dataset files found. You may have forgotten to run the `prep_data.py` preparation process on the dataset you want to use.")

    # collate these into a single data object
    token_data = []
    for d_set in dataset_paths:
        with open(d_set, 'rb') as f:
            data_slice = np.load(f, allow_pickle=True)

        token_data.extend(data_slice.tolist())
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
    if dataset == None:
        return "None"

    calcs = {}
    for i in dataset:
        for tok in i:
            if tok[2] not in calcs.keys():
                calcs[tok[2]] = 1
            else:
                calcs[tok[2]] += 1

    return calcs


def train_model(train_data_txt='rpunct_train_set.txt', val_data_txt='rpunct_val_set.txt', use_cuda=True, validation=True, epochs=3):
    """
    Trains simpletransformers model.
    Args:
        - train_data_txt (str): File name where training dataset is stored (txt in Connl NER format).
        - val_data_txt (str): File name where validation dataset is stored (txt in Connl NER format).
        - use_cuda (bool): Toggle to run training on GPU (True) or CPU (False).
        - validation (bool): Toggle to exploit validation set during training (validates performance every 5000 steps).
    """
    # Create a NERModel
    print("\t* Building NER model")
    model = NERModel(
        "bert",
        "bert-base-uncased",
        labels=VALID_LABELS,
        use_cuda=use_cuda,
        args={
            "evaluate_during_training": validation,
            "evaluate_during_training_steps": 5000,
            "overwrite_output_dir": True,
            "num_train_epochs": epochs,
            "max_seq_length": 512,
            "lazy_loading": True,
            "save_steps": -1,
            "save_model_every_epoch": True
        }
    )

    # Train the model
    train_data_path = os.path.join(PATH, train_data_txt)
    val_data_path = os.path.join(PATH, val_data_txt)
    print(f"\t* Training model on dataset: {train_data_path}")
    print(f"\t* Validate model during training: {validation}")

    steps, tr_details = model.train_model(train_data_path, eval_data=val_data_path)

    return model, steps, tr_details


def plot_training(training, out_path='training/training_loss.png'):
    """
    Plots the progression of the loss function (validation metrics) computed at each 5000 steps during training.
    Args:
        - training (dict): Global step count, and results of metric evalidations.
        - out_path (str): The output file to which the generated training plot is saved.
    """
    palette = sns.color_palette("Set2")
    fig, ax = plt.subplots(1, 1)
    keys = list(training.keys())
    keys.remove('global_step')
    count = 0

    for key in keys:
        sns.lineplot(ax=ax, x='global_step', y=key, data=training, color=palette[count % len(palette)])
        count += 1

    ax.set(ylabel='Loss', xlabel='Training step', title="Progression of Model Training")
    ax.legend(labels=keys, title="Legend")
    fig.savefig(out_path)


if __name__ == "__main__":
    # specify which training data to use and whether to use a GPU
    data = 'news'
    cuda = False

    # run training pipeline using news data
    e2e_train(data_type=data, use_cuda=cuda)
