# Noah-Manuel Michael
# Created: 2025-04-18
# Updated: 2025-06-29
# Utility function for data loading for model training

import pandas as pd
import os
from collections import defaultdict


def load_relevant_train_datasets(configuration, language):
    """
    Loads and returns training and development datasets according to the specified configuration and language.

    :param str configuration: Specifies the data loading strategy.
        Options include: 'target', 'all', 'all-balanced', 'west', 'west-balanced',
        'north', 'north-balanced', 'island', 'island-balanced',
        'mainland', 'mainland-balanced', 'random', 'adjacent'.
    :param str language: The target language code (e.g., 'de', 'nl', etc.).
    :return: tuple (train_df, dev_df) where both are pandas DataFrames containing the training and development data.
    """
    # Determine dataset directory based on configuration
    if configuration in ['target', 'all', 'all-balanced', 'west', 'west-balanced', 'north', 'north-balanced', 'island',
                         'island-balanced', 'mainland', 'mainland-balanced']:
        directory = 'verb_error_datasets_v1_1'
    elif configuration in ['random', 'adjacent']:
        directory = 'verb_error_datasets_naive'
    else:
        raise ValueError('Invalid configuration')

    # Define language groups based on configuration
    if 'all' in configuration:
        lang_list = ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv']
    elif 'west' in configuration:
        lang_list = ['af', 'de', 'nl']
    elif 'north' in configuration:
        lang_list = ['da', 'fo', 'is', 'nb', 'nn', 'sv']
    elif 'island' in configuration:
        lang_list = ['fo', 'is']
    elif 'mainland' in configuration:
        lang_list = ['da', 'nb', 'nn', 'sv']

    # Initialize dataset containers
    train_datasets = []
    dev_datasets = []

    # Train one model for all languages per language group
    if configuration in ['all', 'west', 'north', 'island', 'mainland']:  # load all data from all languages in the respective group
        for lang in lang_list:
            for file in os.listdir(f'{directory}/{lang}'):
                if 'train' in file:
                    train_datasets.append(
                        pd.read_csv(f'{directory}/{lang}/' + file, sep='\t', encoding='utf-8', index_col=0))
                elif 'dev' in file:
                    dev_datasets.append(
                        pd.read_csv(f'{directory}/{lang}/' + file, sep='\t', encoding='utf-8', index_col=0))

        train_df = pd.concat(train_datasets, axis=0)
        train_df = train_df.sample(frac=1, random_state=667)

        dev_df = pd.concat(dev_datasets, axis=0)
        dev_df = dev_df.sample(frac=1, random_state=667)

    # Train one model for each language, add as much data from related languages as possible without exceeding the number of sentences of the target language
    elif configuration in ['all-balanced', 'west-balanced', 'north-balanced', 'island-balanced', 'mainland-balanced']:  # load data from all languages in the respective group but balance according to target language

        train_datasets_target = []
        dev_datasets_target = []
        train_datasets_per_lang = defaultdict(list)
        dev_datasets_per_lang = defaultdict(list)

        for lang in lang_list:
            for file in os.listdir(f'{directory}/{lang}'):
                if lang != language:  # if current language is not target language, store dataframe in dict for each language
                    if 'train' in file:
                        train_datasets_per_lang[lang].append(
                            pd.read_csv(f'{directory}/{lang}/' + file, sep='\t', encoding='utf-8', index_col=0))
                    elif 'dev' in file:
                        dev_datasets_per_lang[lang].append(
                            pd.read_csv(f'{directory}/{lang}/' + file, sep='\t', encoding='utf-8', index_col=0))

                else:  # if current language is target language, store in separate list
                    if 'train' in file:
                        train_datasets_target.append(
                            pd.read_csv(f'{directory}/{lang}/' + file, sep='\t', encoding='utf-8', index_col=0))
                    elif 'dev' in file:
                        dev_datasets_target.append(
                            pd.read_csv(f'{directory}/{lang}/' + file, sep='\t', encoding='utf-8', index_col=0))

        # Concatenate and determine max lengths for balancing
        if len(train_datasets_target) > 1:
            train_df_target = pd.concat(train_datasets_target, axis=0)
            max_len_train = train_df_target.shape[0]
        else:
            train_df_target = train_datasets_target[0]
            max_len_train = train_df_target.shape[0]

        if len(dev_datasets_target) > 1:
            dev_df_target = pd.concat(dev_datasets_target, axis=0)
            max_len_dev = dev_df_target.shape[0]
        else:
            dev_df_target = dev_datasets_target[0]
            max_len_dev = dev_df_target.shape[0]

        # Balance and concatenate non-target language datasets
        for lang, dataframes in train_datasets_per_lang.items():
            if len(dataframes) > 1:
                train_df_per_lang = pd.concat(dataframes, axis=0)
                if train_df_per_lang.shape[0] > max_len_train:
                    train_df_per_lang = train_df_per_lang.sample(max_len_train, random_state=667)
                train_df_target = pd.concat([train_df_target, train_df_per_lang], axis=0)
            else:
                train_df_per_lang = dataframes[0]
                if train_df_per_lang.shape[0] > max_len_train:
                    train_df_per_lang = train_df_per_lang.sample(max_len_train, random_state=667)
                train_df_target = pd.concat([train_df_target, train_df_per_lang], axis=0)

        train_df = train_df_target.sample(frac=1, random_state=667)

        for lang, dataframes in dev_datasets_per_lang.items():
            if len(dataframes) > 1:
                dev_df_per_lang = pd.concat(dataframes, axis=0)
                if dev_df_per_lang.shape[0] > max_len_dev:
                    dev_df_per_lang = dev_df_per_lang.sample(max_len_dev, random_state=667)
                dev_df_target = pd.concat([dev_df_target, dev_df_per_lang], axis=0)
            else:
                dev_df_per_lang = dataframes[0]
                if dev_df_per_lang.shape[0] > max_len_dev:
                    dev_df_per_lang = dev_df_per_lang.sample(max_len_dev, random_state=667)
                dev_df_target = pd.concat([dev_df_target, dev_df_per_lang], axis=0)

        dev_df = dev_df_target.sample(frac=1, random_state=667)

    # Train one model per language, only on data of that language
    elif configuration in ['target', 'random', 'adjacent']:  # load data from the target language only
        for file in os.listdir(f'{directory}/{language}'):
            if 'train' in file:
                train_datasets.append(pd.read_csv(f'{directory}/{language}/' + file, sep='\t', encoding='utf-8', index_col=0))
            elif 'dev' in file:
                dev_datasets.append(pd.read_csv(f'{directory}/{language}/' + file, sep='\t', encoding='utf-8', index_col=0))

        if len(train_datasets) > 1:
            train_df = pd.concat(train_datasets, axis=0)
            train_df = train_df.sample(frac=1, random_state=667)
        else:
            train_df = train_datasets[0]

        if len(dev_datasets) > 1:
            dev_df = pd.concat(dev_datasets, axis=0)
            dev_df = dev_df.sample(frac=1, random_state=667)
        else:
            dev_df = dev_datasets[0]

    # Return the final training and development DataFrames
    return  train_df, dev_df
