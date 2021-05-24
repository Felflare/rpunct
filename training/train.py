# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import json
from simpletransformers.ner import NERModel

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']


def e2e_train():
    prepare_data()
    steps, tr_details = train_model()
    print(f"Steps: {steps}; Train details: {tr_details}")


def train_model():
    """
    Trains simpletransformers model
    """
    # Create a NERModel
    model = NERModel("bert", "bert-base-uncased",
                     args={"overwrite_output_dir": True,
                           "num_train_epochs": 3,
                           "max_seq_length": 512,
                           "lazy_loading": True},
                     labels=VALID_LABELS)

    # # Train the model
    steps, tr_details = model.train_model('rpunct_train_set.txt')
    return steps, tr_details


def prepare_data():
    """
    Prepares data from Original text into Connnl formatted datasets ready for training
    In addition constraints label space to only labels we care about
    """
    token_data = load_datasets(['telp_train_1.txt', 'telp_train_2.txt', 'telp_train_3.txt', 'telp_train_4.txt'])
    clean_up_labels(token_data, valid_labels)
    eval_set = token_data[-int(len(token_data) * 0.10):]
    train_set = token_data[:int(len(token_data) * 0.90)]
    create_text_file(train_set, 'rpunct_train_set.txt')
    create_text_file(eval_set, 'rpunct_test_set.txt')


def load_datasets(dataset_paths):
    """
    Given a list of data paths returns a single data object containing all data slices
    """
    token_data = []
    for d_set in [dataset_paths]:
        with open(d_set, 'r') as fp:
            data_slice = json.load(fp)
        token_data.extend(data_slice)
        del data_slice
    return token_data


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
    print(calcs)
    return calcs


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


if __name__ == "__main__":
    print("Training the model.")
    e2e_train()
