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

VALID_LABELS = ['OC', 'OO', '.O', '!O', ',O', '.C', '!C', ',C', ':O', ';O', ':C', "'O", '-O', '?O', '?C']

PATH = './training/datasets/'
RESULTS_PATH = './tests/'


def e2e_test(models, data_source='reviews', use_cuda=True, print_stats=False, output_file='model_performance.png'):
    """
    Testing model performance after full training process has been completed.
    """
    # format testing data into txt
    prepare_data(source=data_source, print_stats=print_stats, train_or_test='test', validation=False)
    all_metrics = []
    count = 1

    for model_path in models:
        # load fully trained model
        model = NERModel(
            "bert",
            model_path,
            labels=VALID_LABELS,
            use_cuda=use_cuda,
            args={"max_seq_length": 512}
        )
        print(f"\n> Model {count}: {model_path}", end='\n\n')
        count += 1

        # test model after its been fully trained
        test_data_txt = os.path.join(PATH, data_source, 'rpunct_test_set.txt')
        metrics, _, _ = model.eval_model(test_data_txt, output_dir=RESULTS_PATH)
        print(f"\n\t* Results: {metrics}")
        del model
        all_metrics.append(metrics)
        del metrics

    compare_models(all_metrics, models, out_png=output_file, data_type=data_source)

    print("\n> Model testing complete", end='\n\n')


def compare_models(results, model_locations, out_png='model_performance.png', data_type='news'):
    plot_path = os.path.join(RESULTS_PATH, out_png)  # output file to plot to
    df = pd.DataFrame(columns = ['Metrics', 'Results', 'Model'])  # dataframe for storing metrics to be plotted

    # construct dataframe enumerating performance metrics for all models being compared
    count = 0
    for result in results:
        # construct a name for the  model
        model_name_loc = model_locations[count].rfind('/')
        if model_name_loc == -1:
            model_name = model_locations[count]
        else:
            model_name = model_locations[count][model_name_loc + 1:]

        # add model's metrics to the dataframe
        df2 = pd.DataFrame({
            'Metrics': [key.replace('_', ' ').capitalize() for key in result.keys()],
            'Results': result.values(),
            'Model': model_name
        })

        df = pd.concat([df, df2])
        count += 1

    # plot & save single bar chart of each metric & model
    fig, ax = plt.subplots(1, 1)
    sns.barplot(ax=ax, x='Metrics', y='Results', hue='Model', data=df)
    ax.set(title=f"Test Performance of Optimised Models ({data_type} data)")

    fig.savefig(plot_path)
    print(f"\n> Performance comparison saved to: {plot_path}")


if __name__ == "__main__":
    # specify which models to test and what test dataset to use
    data = 'news'
    models = ['outputs/best_model', 'felflare/bert-restore-punctuation']

    # run testing pipeline
    e2e_test(models, data_type=data, use_cuda=False)
