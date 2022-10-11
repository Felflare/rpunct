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
DATA_PATH = './training/datasets/'
RESULTS_PATH = './tests/'


def e2e_test(models, data_type='reviews', use_cuda=True):
    """
    Testing model performance after full training process has been completed.
    """
    # format testing data into txt
    print("\nPreparing testing data")
    test_data_txt = 'rpunct_test_set.txt'
    prepare_data(data_type=data_type, train_or_test='test', validation=False)
    all_metrics = []

    for model_path in models:
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
        all_metrics.append(metrics)

    compare_models(all_metrics)


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


def compare_models(models, out_png='model_performance.png'):
    plot_path = os.path.join(RESULTS_PATH, out_png)
    df = pd.DataFrame(columns = ['Metrics', 'Results', 'Model'])

    count = 1
    for model in models:
        df2 = pd.DataFrame({
            'Metrics': [key.replace('_', ' ').capitalize() for key in model.keys()],
            'Results': model.values(),
            'Model': f"Model {count}"
        })

        df = pd.concat([df, df2])
        count += 1

    fig, ax = plt.subplots(1, 1)
    sns.barplot(ax=ax, x='Metrics', y='Results', hue='Model', data=df)
    ax.set(title="Test Performance of Optimised Models")

    fig.savefig(plot_path)
    print(f"\nPerformance comparison saved to: {plot_path}")


if __name__ == "__main__":
    # read in which models to test and what test dataset to use
    data = sys.argv[1]
    if data == 'news':
        print(f"\nTesting model on data from source: BBC News")
    elif data == 'reviews':
        print(f"\nTesting model on data from source: Yelp reviews")
    else:
        raise ValueError('Unknown data source')

    models = sys.argv[2:]  # likely 'outputs/best_model' and/or 'felflare/bert-restore-punctuation'
    if len(models) == 0:
        raise ValueError('No test models specified')

    e2e_test(models, data_type=data, use_cuda=False)
