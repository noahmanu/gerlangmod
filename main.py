# Noah-Manuel Michael
# Created: 2025-02-14
# Updated: 2025-07-01
# Script to create verb error data

import os
import random
from Datasets.UD_phraser_shuffler_v1_1 import Phraser, Shuffler
from Datasets.UD_simple import Simple
from Experiments.dataloading_utils import load_relevant_train_datasets
from Experiments.finetune_v1_1 import train_model


if __name__ == '__main__':
    random.seed(667)

    # Create GermDetect datasets from UD data
    for directory in os.listdir('Germanic UD'):
        if os.path.isdir('Germanic UD/' + directory):
            for file in os.listdir('Germanic UD/' + directory):
                if file.endswith('.conllu'):
                    phraser = Phraser('Germanic UD/' + directory + '/' + file)
                    shuffler = Shuffler('Germanic UD/' + directory + '/' + file, phraser.noun_phrases_pphr_ps,
                                        phraser.df_data)

    # Create naive baselines from UD data
    for directory in os.listdir('Germanic UD'):
        if os.path.isdir('Germanic UD/' + directory):
            for file in os.listdir('Germanic UD/' + directory):
                if file.endswith('.conllu'):
                    simple = Simple('Germanic UD/' + directory + '/' + file)

    # Train models on GermDetect data
    languages_per_conf = {
        'target': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
        'random': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
        'adjacent': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
        'all': ['all_langs'],  # placeholder; language selection happens in load_relevant_train_datasets
        'all-balanced': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
        'west': ['all_langs'],
        'west-balanced': ['af', 'de', 'nl'],
        'north': ['all_langs'],
        'north-balanced': ['da', 'fo', 'is', 'nb', 'nn', 'sv'],
        'island': ['all_langs'],
        'island-balanced': ['fo', 'is'],
        'mainland': ['all_langs'],
        'mainland-balanced': ['da', 'nb', 'nn', 'sv'],
    }

    for config, langs in languages_per_conf.items():
        for lang in langs:
            train_df, dev_df = load_relevant_train_datasets(config, lang)
            train_model(train_df, dev_df, config, lang)
