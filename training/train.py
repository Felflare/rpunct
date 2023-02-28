# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
from simpletransformers.ner import NERModel
import matplotlib.pyplot as plt
import seaborn as sns
from training.prep_data import VALID_LABELS

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(10, 7), 'figure.dpi':100, 'savefig.dpi':100})

PATH = './training/datasets/'


def e2e_train(
            model_source=None, data_source='news-transcripts',
            epochs=3, use_cuda=True,
            validation=False, training_plot=False):
    """
    Full pipeline for building (or loading) and training a transformer-based RPunct model (using previously prepared RPunct dataset)
    for the punctuation recovery task.

    Args:
        - model_source (str): the name of the directory within `rpunct/outputs/` to load a pre-trained RPunct model from (or create a new model if None).
        - data_source (str): the name of the directory within `rpunct/training/datasets/` containing the prepared training data.
        - epochs (int): the number of training epochs to execute.
        - use_cuda (bool): run training on GPU (True) or CPU (False).
        - validation (bool): use a pre-prepared validation set during training (validates performance every 5000 steps).
        - training_plot (bool): generate an output plot of the progression of the validation loss (requires `validation = True`)
        every 5000 steps during training.
    """
    # Create or load a simpletransformers NERModel to be the RPunct model
    if model_source is None:
        print("\t* Building RPunct model", end='\n\n')
        model = NERModel(
            "bert",
            "bert-base-uncased",
            labels=VALID_LABELS,
            use_cuda=use_cuda,
            args={
                "evaluate_during_training": validation,
                "evaluate_during_training_steps": 5000,
                "evaluate_during_training_verbose": True,
                "save_eval_checkpoints": False,
                "overwrite_output_dir": True,
                "num_train_epochs": epochs,
                "max_seq_length": 512,
                "lazy_loading": True,
                "save_steps": -1,
                "save_model_every_epoch": True,
                "scheduler": 'polynomial_decay_schedule_with_warmup'
            }
        )
    else:
        print("\t* Loading RPunct model")
        model_location = os.path.join('outputs', model_source)

        if os.path.isdir(model_location):
            model = NERModel(
                "bert",
                model_location,
                use_cuda=use_cuda,
                args={
                    "num_train_epochs": epochs
                }
            )
            print(f"\t* Pre-trained model found at: {model_location}", end='\n\n')
        else:
            raise FileNotFoundError(f"Pre-trained model files could not be found at the path: {model_location}")

    # Train the model
    train_data_path = os.path.join(PATH, data_source, 'rpunct_train_set.txt')
    val_data_path = os.path.join(PATH, data_source, 'rpunct_val_set.txt')
    print(f"\n\t* Training model on dataset: {train_data_path}")
    print(f"\t* Validate model during training: {validation}", end='\n\n')

    steps, tr_details = model.train_model(train_data_path, eval_data=val_data_path)
    print(f"\n\t* Steps: {steps}; Train details: {tr_details}")

    # Plot the progression/convergence over training/validation
    if validation and training_plot:
        plot_training(tr_details)

    print("\n> Model training complete", end='\n\n')

    return model


def plot_training(training, out_path='training/training_loss.png'):
    """
    Plots the progression of the loss function (validation metrics) computed at each 5000 steps during training.
    Args:
        - training (dict): global step count and results of metric evalidations for all complete training epochs.
        - out_path (str): the output file path (from `rpunct/`) to save the generated training plot to.
    """
    # Styling and data
    palette = sns.color_palette("Set2")
    fig, ax = plt.subplots(1, 1)
    keys = list(training.keys())
    keys.remove('global_step')
    count = 0

    # Plotting
    for key in keys:
        sns.lineplot(ax=ax, x='global_step', y=key, data=training, color=palette[count % len(palette)])
        count += 1

    ax.set(ylabel='Loss', xlabel='Training step', title="Progression of Model Training")
    ax.legend(labels=keys, title="Legend")
    fig.savefig(out_path)


if __name__ == "__main__":
    # Specify which training data to use and whether to use a GPU
    data = 'news-transcripts'
    cuda = False

    # Run training pipeline using news data
    e2e_train(data_type=data, use_cuda=cuda)
