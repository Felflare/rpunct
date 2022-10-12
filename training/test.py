# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Tom Potter"
__email__ = "tom.potter@bbc.co.uk"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simpletransformers.ner import NERModel

try:
    from training.train import prepare_data
except ModuleNotFoundError:
    from train import prepare_data

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(10, 7), 'figure.dpi':100, 'savefig.dpi':100})

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
DATA_PATH = './training/datasets/'
RESULTS_PATH = './tests/'


def e2e_test(models, data_type='reviews', use_cuda=True, print_stats=False):
    """
    Testing model performance after full training process has been completed.
    """
    # format testing data into txt
    print("\nPreparing testing data")
    test_data_txt = 'rpunct_test_set.txt'
    prepare_data(data_type=data_type, print_stats=print_stats, train_or_test='test', validation=False)
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

    compare_models(all_metrics, models)


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


def compare_models(results, model_locations, out_png='model_performance.png'):
    plot_path = os.path.join(RESULTS_PATH, out_png)
    df = pd.DataFrame(columns = ['Metrics', 'Results', 'Model'])

    count = 0
    for result in results:
        df2 = pd.DataFrame({
            'Metrics': [key.replace('_', ' ').capitalize() for key in result.keys()],
            'Results': result.values(),
            'Model': f"Model {count}: {model_locations[count]}"
        })

        df = pd.concat([df, df2])
        count += 1

    fig, ax = plt.subplots(1, 1)
    sns.barplot(ax=ax, x='Metrics', y='Results', hue='Model', data=df)
    ax.set(title="Test Performance of Optimised Models")

    fig.savefig(plot_path)
    print(f"\nPerformance comparison saved to: {plot_path}")


if __name__ == "__main__":
    # specify which models to test and what test dataset to use
    data = 'news'
    models = ['outputs/best_model', 'felflare/bert-restore-punctuation']

    # run testing pipeline
    e2e_test(models, data_type=data, use_cuda=False)
