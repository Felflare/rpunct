# -*- coding: utf-8 -*-
# 💾⚙️🔮

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simpletransformers.ner import NERModel
from train import prepare_data

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(10, 7), 'figure.dpi':100, 'savefig.dpi':100})


VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
TEST_DATASETS = ['yelp_test_1.txt', 'yelp_test_2.txt', 'yelp_test_3.txt', 'yelp_test_4.txt']
DATA_PATH = './training/datasets/'
RESULTS_PATH = './tests/'


def e2e_test(model_path, results_txt='rpunct_test_results.txt', use_cuda=True):
    """
    Testing model performance after full training process has been completed.
    """
    # format testing data into txt
    print("\nPreparing testing data")
    test_data_txt = 'rpunct_test_set.txt'
    test_data_lst = prepare_data(TEST_DATASETS, output_txt=test_data_txt, validation=False)

    # load fully trained model
    model = NERModel(
        "bert",
        model_path,
        labels=VALID_LABELS,
        use_cuda=use_cuda,
        args={"max_seq_length": 512}
    )
    print(f"\nModel loaded from: {model_path}")

    # test model after its been fully trained
    metrics, outputs, predictions = test_model(model, test_data_txt)

    compare_models(metrics, metrics)


def test_model(model, in_txt):
    """
    Use prepaered test dataset in txt file `in_txt` to test the optimised model `model`
    and write the results to the output directory `RESULTS_PATH`.
    Args:
        - model (str): Location where optimised parameterisation of model is stored.
        - in_txt (str): File name of testing dataset (within datasets directory).
    """
    # load data and test model
    test_data_path = os.path.join(DATA_PATH, in_txt)
    print(f"\nTesting model on dataset: {test_data_path}", end='\n\n')

    result, model_outputs, wrong_preds = model.eval_model(test_data_path, output_dir=RESULTS_PATH)
    print(f"\nResults: {result}")

    return result, model_outputs, wrong_preds


def compare_models(model1, model2=None):
    df = pd.DataFrame({
       'metrics': model1.keys(),
       'results': model1.values(),
       'model': 'model1'
    })

    if model2 is not None:
        df2 = pd.DataFrame({
            'metrics': model2.keys(),
            'results': model2.values(),
            'model': 'model2'
        })

        df = pd.concat([df, df2])

    fig, ax = plt.subplots(1, 1)
    sns.barplot(ax=ax, x='metrics', y='results', hue='model', data=df)
    fig.savefig('bar_chart.png')


if __name__ == "__main__":
    # take input of model's optimised parameterisation from command line (if none use hugging face model)
    inputs = sys.argv[1:]
    if len(inputs) == 0:
        model = 'felflare/bert-restore-punctuation'
    else:
        model = inputs[0]  # likely 'outputs/best_model'

    e2e_test(model, 'orig_rpunct_test_results.txt', use_cuda=True)
