import os
import json
import math
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

PATH = './training/datasets/'
INTERVIEW_PATH = '/Users/tompo/.bbc-data/bbc/speech/transcription/evaluation/bbc-oral-history-project/0.0.1/data/original-metadata/GoldStandardNC'


def remove_temp_files(dataset_path, extensions=['npy', 'csv'], traintest=''):
    if traintest != '':
        traintest = '*' + traintest

    # find directory and remove all temporary .npy and .csv files
    for extension in extensions:
        for p in pathlib.Path(dataset_path).glob(f"{traintest}*." + extension):
            p.unlink()


def create_composite_dataset(distinct, train_split, dataset_names, balance):
    # create a collection of all individual datasets needed to construct composite datasets
    all_datasets = []

    for name in dataset_names:
        # collect dataset from file
        if name == 'news-articles':
            # collect articles part of composite dataset (from JSONL files)
            dataset_dir = collate_news_articles(2019, 2022, summaries=False, train_split=1.0, composite=True)
            dataset_path = os.path.join(dataset_dir, 'train_news.csv')

        elif name == 'news-transcripts':
            # collect transcripts part of dataset
            dataset_dir = collate_news_transcripts(train_split=0.9998, composite=True)
            dataset_path = os.path.join(dataset_dir, 'train_transcripts.csv')

            # only construct testing data from transcripts dataset
            test_dataset_path_input = os.path.join(dataset_dir, 'test_transcripts.csv')
            transcripts_test = pd.read_csv(test_dataset_path_input)
            transcripts_test.dropna(inplace=True)
            transcripts_test.reset_index(drop=True, inplace=True)

            test_dataset_path_output = os.path.join(dataset_dir, 'test_composite.csv')
            transcripts_test.to_csv(test_dataset_path_output, index=False)
            del transcripts_test

        elif name == 'historical-interviews':
            # collect historical interviews part of dataset
            dataset_dir = collate_historical_interviews(composite=True)
            dataset_path = os.path.join(dataset_dir, 'train_interviews.csv')

        else:
            raise ValueError("Composite dataset cannot be built with unknown source datasets.")

        # format dataset
        dataset = pd.read_csv(dataset_path)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        # if wanting to create distinct composite dataset (i.e. separate pre-train/tune) then label each dataset to be split up later
        if distinct:
            dataset['source'] = name

        # add to dataset collection
        all_datasets.append(dataset.copy())
        del dataset

    # combine two news datasets together in proportion denoted by `balance`
    # if balance == '1:1':  # even balance
    #     min_length = min(map(len, all_datasets))
    #     all_datasets = [d[:min_length] for d in all_datasets]
    #     print("\n> Combining datasets in 1:1 proportion")
    # elif balance == '2:1' and len(all_datasets) == 2:
    #     proportion = int(len(all_datasets[0]) / 2)

    #     if len(all_datasets[1]) > proportion and len(all_datasets) == 2:
    #         all_datasets = [all_datasets[0], all_datasets[1][:proportion]]
    #         print("\n> Combining datasets in 2:1 proportion")
    #     else:
    #         print("\n> Combining datasets of original length")
    # elif balance == '1:2' and len(all_datasets) == 2:
    #     proportion = int(len(all_datasets[1]) / 2)

    #     if len(all_datasets[0]) > proportion:
    #         all_datasets = [all_datasets[0][:proportion], all_datasets[1]]
    #         print("\n> Combining datasets of in 1:2 proportion")
    #     else:
    #         print("\n> Combining datasets of original length")
    # elif balance == '2:3' and len(all_datasets) == 2:
    #     proportion = int(len(all_datasets[1]) / 3) * 2

    #     if len(all_datasets[0]) > proportion:
    #         all_datasets = [all_datasets[0][:proportion], all_datasets[1]]
    #         print("\n> Combining datasets of in 2:3 proportion")
    #     else:
    #         print("\n> Combining datasets of original length")
    # else:  # `balance = 'o'` => original dataset sizes
    #     print("\n> Combining datasets of original length")

    composite_data = pd.concat(all_datasets, ignore_index=True)
    del all_datasets

    # shuffle samples (not necessary if splitting into distinct composite datasets)
    if not distinct:
        composite_data = composite_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save composite dataset to csv file
    csv_path_train = os.path.join(dataset_dir, 'train_composite.csv')
    composite_data.to_csv(csv_path_train, index=False)
    del composite_data

    return dataset_dir


def collate_news_articles(start_date, end_date, summaries, train_split=0.9, composite=False):
    if composite:
        summary_or_body = 'body'
        dataset_path = os.path.join(PATH, 'composite')
    elif summaries:
        summary_or_body = 'summary'
        dataset_path = os.path.join(PATH, f'news-summaries')
    else:
        summary_or_body = 'body'
        dataset_path = os.path.join(PATH, f'news-{start_date}-{end_date}')

    print(f"\n> Assembling news article {summary_or_body[:-1]}ies:")
    news_datasets = [f'news_{date}.jsonl' for date in range(start_date, end_date + 1)]
    articles = []

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'news_data_2014-22/', dataset_json)

        with open(json_path, 'r') as fp:
            for line in fp:
                obj = json.loads(line)
                articles.append(obj[summary_or_body])
                del obj

    # train-test split
    random.seed(42)
    random.shuffle(articles)
    split = math.ceil(train_split * len(articles))
    train = articles[:split]
    test = articles[split:]

    print(f"\t* Articles in total    : {len(articles)}")
    print(f"\t* Articles in train set: {len(train)}")
    print(f"\t* Articles in test set : {len(test)}")
    del articles

    # save train/test data to csv
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n', ' ')
    csv_path_train = os.path.join(dataset_path, 'train_news.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(dataset_path, 'test_news.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    return dataset_path


def collate_news_transcripts(train_split=0.9, composite=False):
    # input transcripts from json files
    print(f"\n> Assembling news transcripts:")
    news_datasets = ['transcripts_2014-17.json', 'transcripts_2020.json']
    transcripts = np.empty(shape=(0), dtype=object)

    for dataset_json in news_datasets:
        json_path = os.path.join(PATH, 'news_transcripts_2014-20/', dataset_json)

        with open(json_path, 'r') as f:
            obj = json.load(f)

        data = pd.DataFrame(obj["Transcripts"])
        speaker_segments = np.concatenate(data["Items"]).flat
        transcripts = np.append(transcripts, speaker_segments)

        del obj
        del data
        del speaker_segments

    # train-test split
    random.seed(42)
    random.shuffle(transcripts)
    split = math.ceil(train_split * len(transcripts))
    train = transcripts[:split]
    test = transcripts[split:]

    print(f"\t* {train_split:.1f} : {1-train_split:.1f} data split")
    print(f"\t* Speaker segments in total    : {len(transcripts)}")
    print(f"\t* Speaker segments in train set: {len(train)}")
    print(f"\t* Speaker segments in test set : {len(test)}")
    del transcripts

    # save train/test data to csv
    if composite:
        dataset_path = os.path.join(PATH, 'composite')
    else:
        dataset_path = os.path.join(PATH, f'news-transcripts')

    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n',' ')
    csv_path_train = os.path.join(dataset_path, 'train_transcripts.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(dataset_path, 'test_transcripts.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    return dataset_path


def download_reviews():
    # download yelp reviews data from tensorflow_datasets
    train, test = tfds.load('yelp_polarity_reviews', split=['train', 'test'], shuffle_files=True)

    # train-test split
    train = tfds.as_dataframe(train)
    train['text'] = train['text'].str.decode("utf-8")

    test = tfds.as_dataframe(test)
    test['text'] = test['text'].str.decode("utf-8")

    # filter to only positive examples
    train = train[train['label'] == 1].reset_index(drop=True)
    train = train.drop(columns=['label'])
    test = test[test['label'] == 1].reset_index(drop=True)
    test = test.drop(columns=['label'])

    # save train/test data to csv
    dataset_path = os.path.join(PATH, 'reviews')
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    csv_path_train = os.path.join(dataset_path, 'train_reviews.csv')
    train.to_csv(csv_path_train, index=False)

    csv_path_test = os.path.join(dataset_path, 'test_reviews.csv')
    test.to_csv(csv_path_test, index=False)

    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    return dataset_path


def collate_historical_interviews(train_split=1.0, composite=False):
    print(f"\n> Assembling interview transcripts:")

    if composite:
        dataset_path = os.path.join(PATH, 'composite')
    else:
        dataset_path = os.path.join(PATH, 'historical-interview-transcripts')

    dataset_files = [f'{dir}/{dir}.txt' for dir in os.listdir(INTERVIEW_PATH)]
    interviews = []

    for dataset_txt in dataset_files:
        file_path = os.path.join(INTERVIEW_PATH, dataset_txt)

        with open(file_path, 'r') as fp:
            data = np.array(fp.readlines())

            mapping =  dict.fromkeys(range(32))
            data = [d.translate(mapping) for d in data if d not in ['\t', '\n']]

        interviews.extend([d.strip() for d in data if "{" not in d and "\\" not in d])
        del data

    # train-test split
    random.seed(42)
    random.shuffle(interviews)
    split = math.ceil(train_split * len(interviews))
    train = interviews[:split]
    test = interviews[split:]

    print(f"\t* Interview transcripts in total    : {len(interviews)}")
    print(f"\t* Interview transcripts in train set: {len(train)}")
    print(f"\t* Interview transcripts in test set : {len(test)}")
    del interviews

    # save train/test data to csv
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace("_", '')
    train['text'] = train['text'].replace('', np.nan)
    train['text'] = train['text'].dropna()

    csv_path_train = os.path.join(dataset_path, 'train_interviews.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace("_", '')
    test['text'] = test['text'].replace('', np.nan)
    test['text'] = test['text'].dropna()

    csv_path_test = os.path.join(dataset_path, 'test_interviews.csv')
    test.to_csv(csv_path_test, index=False)
    del test

    # remove pre-existing data files from previous iterations
    remove_temp_files(dataset_path, extensions=['npy', 'txt'])

    return dataset_path
