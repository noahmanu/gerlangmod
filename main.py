# Noah-Manuel Michael
# Created: 2025-02-14
# Updated: 2025-02-14
# Script to create verb error data

import os
from UD_phraser_shuffler import Phraser, Shuffler

if __name__ == '__main__':
    for directory in os.listdir('Germanic UD'):
        if os.path.isdir('Germanic UD/' + directory):
            for file in os.listdir('Germanic UD/' + directory):
                if file.endswith('.conllu'):
                    phraser = Phraser('Germanic UD/' + directory + '/' + file)
                    shuffler = Shuffler('Germanic UD/' + directory + '/' + file, phraser.noun_phrases_pphr_ps, phraser.df_data)
