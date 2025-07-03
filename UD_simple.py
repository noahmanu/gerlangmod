# Noah-Manuel Michael
# Created: 2025-03-18
# Updated: 2025-04-14
# Class for creating naive verb error data

import os
import string
import pyconll
import pandas as pd
import random
import re
import copy
# import pprint

# pp = pprint.PrettyPrinter(indent=0)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
# random.seed(667)


class Simple:
    """
    A class that extracts phrases from sentences provided in a conllu file.
    Each phrase is defined as the set of tokens headed by their respective deepest verb token.
    Also extract noun phrases from the previously extracted phrases.
    """
    def __init__(self, conllu_file: str):
        """
        Initialize the class with a conllu file and whether to additionally extract noun phrases from the
        extracted phrases.
        :param str conllu_file: The path to the conllu file to be processed
        """
        self.file = conllu_file
        self.df_data = pd.DataFrame()
        self.sentences = self.get_valid_sentences(self.file)
        self.token_objects = self.swap_adjacent()
        self.random_mess()
        self.save_data()

    def get_valid_sentences(self, conllu_file: str):
        """
        Filter the sentences obtained from the conllu file for the presence of verb tokens and consistency in
        head annotation.
        Return sentences and save original sentences to self.df_data.

        :param str conllu_file: Path to the conllu file
        :return: List of sentences where each sentence is a list of pyconll.unit.token.Token objects
        """
        conllu_sentences = pyconll.load_from_file(conllu_file)

        original_sentences_for_df = []
        sentences = []

        for sentence in conllu_sentences:
            sentence_has_verb = False
            sentence_is_invalid = False
            for word in sentence:
                if word.upos == 'VERB' or word.upos == 'AUX':
                    sentence_has_verb = True
                if word.head is None:
                    sentence_is_invalid = True
            if sentence_has_verb is not True or sentence_is_invalid is True:
                continue

            # Exclude tokens with '-' in their IDs (such as contraction tokens)
            sentence = [word for word in sentence if '-' not in word.id]
            original_sentences_for_df.append(' '.join([word.form for word in sentence]))
            sentences.append(sentence)

        self.df_data['original'] = original_sentences_for_df

        return sentences

    def swap_adjacent(self):
        """
        For each sentence, swap the position of verb or auxiliary tokens with their adjacent token,
        using a controlled heuristic so that some tokens are swapped left and some right.
        After swapping the numeric IDs, the sentence tokens are re-sorted by these IDs to produce the final order.
        """
        no_punc_lower_sentences_for_df = []
        token_objects = []
        adjacent_for_df = []
        adjacent_gold = []

        for sentence in self.sentences:
            # Remove punctuation tokens from the sentence for a lower-case representation.
            sent = [word for word in sentence if word.form not in string.punctuation]
            no_punc_lower_sentences_for_df.append(' '.join(word.form.lower() for word in sent))

            tokens = []
            for word in sent:
                # Label VERB/AUX tokens with 'C' by default, others as 'O'
                label = 'C' if word.upos in ('VERB', 'AUX') else 'O'
                tokens.append([int(word.id), word.form.lower(), word.upos, label])

            token_objects.append(tokens)

        # Make a deep copy for later use, preserving original token order before swaps.
        copy_token_objects = copy.deepcopy(token_objects)

        right_swaps = 0
        left_swaps = 0
        no_swaps = 0
        verbs = 0

        for token_obj in token_objects:
            for i, tok in enumerate(token_obj):
                if tok[2] == 'VERB' or tok[2] == 'AUX':
                    verbs += 1
                    if left_swaps < right_swaps and left_swaps <= no_swaps / 2 and i != 0:
                        cash = token_obj[i - 1][0]  # stores the position of the previous token
                        token_obj[i - 1][0] = tok[0]  # assigns the position of the current verb token to its left neighbor
                        token_obj[i][0] = cash  # assigns the original position of the verb's left neighbor to the verb
                        token_obj[i][3] = 'F'  # marks the verb as incorrect
                        left_swaps += 1
                    elif right_swaps <= left_swaps and right_swaps <= no_swaps / 2 and i+1 < len(token_obj):
                        cash = token_obj[i + 1][0]  # stores the position of the following token
                        token_obj[i + 1][0] = tok[0]  # assigns the position of the current verb token to its right neighbor
                        token_obj[i][0] = cash  # assigns the original position of the verb's right neighbor to the verb
                        token_obj[i][3] = 'F'  # marks the verb as incorrect
                        right_swaps += 1
                    else:
                        no_swaps += 1

            new_order_sent = sorted(token_obj)
            sent = [tok[1] for tok in new_order_sent]
            adjacent_for_df.append(' '.join(sent))
            gold = [tok[3] for tok in new_order_sent]
            adjacent_gold.append(' '.join(gold))

        assert verbs == right_swaps + left_swaps + no_swaps, "Swapping didn't work as expected."

        self.df_data['no_punc_lower'] = no_punc_lower_sentences_for_df
        self.df_data['adjacent'] = adjacent_for_df
        self.df_data['adjacent_gold'] = adjacent_gold

        return copy_token_objects

    def random_mess(self):
        """
        Randomly reposition half of the verb and auxiliary tokens.
        The relative order of non-verb tokens and fixed verbs is preserved.
        Moved verbs are reinserted at random positions and marked with 'F' in the gold labels,
        while tokens that are not moved are labelled with 'O' (non-verbs) or 'C' (fixed verbs).
        """
        random_for_df = []
        random_gold = []

        for sentence in self.token_objects:
            n = len(sentence)
            all_positions = list(range(n))
            sentence_mask = all_positions.copy()
            labels_mask = all_positions.copy()

            movable_tokens = [(i, token) for i, token in enumerate(sentence) if token[2] in ('VERB', 'AUX')]
            num_movable = len(movable_tokens)

            if num_movable % 2 == 0:
                num_to_keep = num_movable // 2
            else:
                num_to_keep = random.choice([num_movable // 2, num_movable // 2 + 1])

            indices_to_keep = set(random.sample(range(num_movable), num_to_keep))

            fixed_movable = []
            moved_movable = []
            for j, (orig_index, token) in enumerate(movable_tokens):
                if j in indices_to_keep:
                    fixed_movable.append((orig_index, token))
                else:
                    moved_movable.append((orig_index, token))

            cash_sentence = []
            cash_labels = []
            for i, token in enumerate(sentence):
                if token[2] in ('VERB', 'AUX'):
                    if any(i == m_orig for m_orig, _ in moved_movable):
                        continue
                    else:
                        cash_sentence.append(token[1])
                        cash_labels.append('C')
                else:
                    cash_sentence.append(token[1])
                    cash_labels.append('O')

            available_positions = all_positions.copy()
            for orig_index, token in moved_movable:
                valid_positions = [pos for pos in available_positions if pos != orig_index]
                if not valid_positions:
                    new_pos = orig_index
                else:
                    new_pos = random.choice(valid_positions)
                available_positions.remove(new_pos)
                sentence_mask[new_pos] = token[1]
                if new_pos == orig_index:
                    labels_mask[new_pos] = 'C'
                else:
                    labels_mask[new_pos] = 'F'

            for i, element in enumerate(sentence_mask):
                if isinstance(sentence_mask[i], int):
                    if cash_sentence:
                        sentence_mask[i] = cash_sentence.pop(0)
                if isinstance(labels_mask[i], int):
                    if cash_labels:
                        labels_mask[i] = cash_labels.pop(0)

            random_for_df.append(" ".join(sentence_mask))
            random_gold.append(" ".join(labels_mask))

        self.df_data['random'] = random_for_df
        self.df_data['random_gold'] = random_gold

    def save_data(self):
        # TODO: Update this so it automatically splits no in nb and nn (for now, I manually created directories for holding these datasets)
        """
        Save the dataframe containing processed phrase data to a TSV file.

        :return: None
        """
        if 'verb_error_datasets_naive' not in os.listdir(os.getcwd()):
            os.mkdir('verb_error_datasets_naive')

        if 'train' in self.file:
            split = 'train'
        elif 'dev' in self.file:
            split = 'dev'
        else:
            split = 'test'

        def extract_language_id(path):
            """
            Helper function to extract the UD dataset identifier from the filepath.

            :param path: Path to the conllu file
            :return: UD dataset identifier
            """
            match = re.search(r'/([^/]+)-ud-[^/]+\.conllu$', path)
            return match.group(1) if match else None

        identifier = extract_language_id(self.file)

        if identifier.split("_")[0] not in os.listdir('verb_error_datasets_naive'):
            os.mkdir(f'verb_error_datasets_naive/{identifier.split("_")[0]}')

        self.df_data.to_csv(
            f'verb_error_datasets_naive/{identifier.split("_")[0]}/{identifier}_{split}_naive.tsv',
            sep='\t',
            encoding='utf-8',
            index_label='idx'
        )
        print(f'{identifier}_{split}_naive.tsv written to disk.')


# Simple('Germanic UD/UD_Afrikaans-AfriBooms/af_afribooms-ud-test.conllu')
