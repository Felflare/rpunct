# -*- coding: utf-8 -*-
# 💾⚙️🔮

import os
from simpletransformers.ner import NERModel
from train import prepare_data


VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
# TEST_DATASETS = ['yelp_test_1.txt', 'yelp_test_2.txt', 'yelp_test_3.txt', 'yelp_test_4.txt']
TEST_DATASETS = ['yelp_test_0.txt']
DATA_PATH = './training/datasets/'
RESULTS_PATH = './tests/'


def e2e_test(model_path, results_txt='rpunct_test_results.txt', use_cuda=True):
    """
    Testing model performance after full training process has been completed.
    """
    # format testing data into txt
    print("\nPreparing testing data")
    test_data_txt = 'rpunct_test_set.txt'
    prepare_data(TEST_DATASETS, output_txt=test_data_txt, validation=False)

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
    test_model(model, test_data_txt)



def test_model(model, in_txt):
    """
    Use prepaered test dataset in txt file `in_txt` to test the optimised model `model`
    and write the results to the output directory `RESULTS_PATH`.
    """
    # load data and test model
    test_data_path = os.path.join(DATA_PATH, in_txt)
    print(f"\nTesting model on dataset: {test_data_path}", end='\n\n')

    result, model_outputs, wrong_preds = model.eval_model(test_data_path, output_dir=RESULTS_PATH)
    print(f"\nResults: {result}")


if __name__ == "__main__":
    model = 'felflare/bert-restore-punctuation'
    e2e_test(model, 'orig_rpunct_test_results.txt', use_cuda=False)
