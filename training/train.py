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


def e2e_train(data_source='reviews', use_cuda=True, validation=False, training_plot=False, epochs=3):
    """
    Full pipeline for building and training a transformer-based RPunct model.

    Args:
        -
    """
    # create a simpletransformer model and use data to train it
    print("\n> Building & training model")
    model, steps, tr_details = train_model(
        data_dir=data_source,
        use_cuda=use_cuda,
        validation=validation,
        epochs=epochs
    )
    print(f"\n\t* Steps: {steps}; Train details: {tr_details}")

    # plot the progression/convergence over training/validation
    if validation and training_plot:
        plot_training(tr_details)

    print("\n> Model training complete", end='\n\n')

    return model


def train_model(model=None, data_dir='reviews', train_data_txt='rpunct_train_set.txt', val_data_txt='rpunct_val_set.txt', use_cuda=True, validation=False, epochs=3):
    """
    Trains simpletransformers model.
    Args:
        - train_data_txt (str): File name where training dataset is stored (txt in Connl NER format).
        - val_data_txt (str): File name where validation dataset is stored (txt in Connl NER format).
        - use_cuda (bool): Toggle to run training on GPU (True) or CPU (False).
        - validation (bool): Toggle to exploit validation set during training (validates performance every 5000 steps).
    """
    # Create a NERModel
    if model is None:
        print("\t* Building NER model", end='\n\n')
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
        print("\t* Loading NER model", end='\n\n')

    # Train the model
    train_data_path = os.path.join(PATH, data_dir, train_data_txt)
    val_data_path = os.path.join(PATH, data_dir, val_data_txt)
    print(f"\n\t* Training model on dataset: {train_data_path}")
    print(f"\t* Validate model during training: {validation}", end='\n\n')

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
