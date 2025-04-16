# Noah-Manuel Michael
# Created: 2025-02-14
# Updated: 2025-04-11
# Script to create verb error data

import os
import random
from UD_phraser_shuffler import Phraser, Shuffler
from UD_simple import Simple

if __name__ == '__main__':
    random.seed(667)

    # for directory in os.listdir('Germanic UD'):
    #     if os.path.isdir('Germanic UD/' + directory):
    #         for file in os.listdir('Germanic UD/' + directory):
    #             if file.endswith('.conllu'):
    #                 phraser = Phraser('Germanic UD/' + directory + '/' + file)
    #                 shuffler = Shuffler('Germanic UD/' + directory + '/' + file, phraser.noun_phrases_pphr_ps, phraser.df_data)

    for directory in os.listdir('Germanic UD'):
        if os.path.isdir('Germanic UD/' + directory):
            for file in os.listdir('Germanic UD/' + directory):
                if file.endswith('.conllu'):
                    simple = Simple('Germanic UD/' + directory + '/' + file)
