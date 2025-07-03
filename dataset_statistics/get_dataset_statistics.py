# Noah-Manuel Michael
# Created: 2025-03-13
# Updated: 2025-06-24
# Get statistics for the created datasets

import os
import pandas as pd
from collections import Counter

# Remove existing statistics file if it exists
if 'dataset_statistics_v1_1.txt' in os.listdir(os.getcwd()):
    os.remove('dataset_statistics_v1_1.txt')

# Initialize counters for all labels and total sentence count
all_labels = Counter()
df_len = 0

# Iterate over all language directories and files in the dataset directory
for directory in os.listdir('../verb_error_datasets_v1_1'):
    for file in os.listdir(os.path.join('../verb_error_datasets_v1_1', directory)):
        counter = Counter()
        # Read the current dataset file
        df = pd.read_csv(os.path.join('../verb_error_datasets_v1_1', directory, file), sep='\t', encoding='utf-8')
        df_len += len(df)
        # Count label occurrences in the permuted_gold column
        for labels in df['permuted_gold']:
            counter.update(labels.split())
            all_labels.update(labels.split())
        # Write per-file statistics to the output file
        with open('dataset_statistics_v1_1.txt', 'a') as f:
            f.write('Language: ' + directory + ', Dataset: ' + file.split('_')[-2] + ', Split: ' + file.split('_')[-1] +
                    ', Total number of sentences: ' + str(len(df)) + '\n' + str(counter) + '\n')

# Write total statistics to the output file
with open('dataset_statistics_v1_1.txt', 'a') as f:
    f.write('Total_sents: ' + str(df_len) + '\n' + str(all_labels))
