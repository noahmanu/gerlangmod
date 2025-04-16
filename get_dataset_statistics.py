# Noah-Manuel Michael
# Created: 2025-03-13
# Updated: 2025-03-13
# Get statistics for the created datasets

import os
import pandas as pd
from collections import Counter

if 'dataset_statistics.txt' in os.listdir(os.getcwd()):
    os.remove('dataset_statistics.txt')

all_labels = Counter()
df_len = 0

for directory in os.listdir('verb_error_datasets'):
    for file in os.listdir(os.path.join('verb_error_datasets', directory)):
        counter = Counter()
        df = pd.read_csv(os.path.join('verb_error_datasets', directory, file), sep='\t', encoding='utf-8')
        df_len += len(df)
        for labels in df['permuted_gold']:
            counter.update(labels.split())
            all_labels.update(labels.split())
        with open('dataset_statistics.txt', 'a') as f:
            f.write('Language: ' + directory + ', Dataset: ' + file.split('_')[-2] + ', Split: ' + file.split('_')[-1] +
                    ', Total number of sentences: ' + str(len(df)) + '\n' + str(counter) + '\n')

with open('dataset_statistics.txt', 'a') as f:
    f.write('Total_sents: ' + str(df_len) + '\n' + str(all_labels))
