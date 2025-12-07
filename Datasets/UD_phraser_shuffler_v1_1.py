# Noah-Manuel Michael
# Created: 2025-01-09
# Updated: 2025-06-28
# Classes for creating verb error data

import os
import string
import pyconll
import pandas as pd
import random
import math
import re
# import pprint

# Pretty printer and pandas display options for debugging (commented out)
# pp = pprint.PrettyPrinter(indent=0)
# pd.set_option("display.max_columns", None)

# The Phraser class is responsible for extracting phrase and noun phrase structures from sentences in a CoNLL-U file
class Phraser:
    """
    A class that extracts phrases from sentences provided in a conllu file.
    Each phrase is defined as the set of tokens headed by their respective deepest verb token.
    Also extract noun phrases from the previously extracted phrases.
    """
    def __init__(self, conllu_file: str):
        """
        Initialize the class with a conllu file and whether to additionally extract noun phrases from the
        extracted phrases.
        Loads the file, filters valid sentences, and extracts phrase/noun phrase structures.
        :param str conllu_file: The path to the conllu file to be processed
        """
        self.file = conllu_file
        self.df_data = pd.DataFrame()
        # Get valid sentences and store them
        self.sentences = self.get_valid_sentences(self.file)
        # Extract dictionaries mapping tokens to their deepest verbal heads
        self.verbal_root_dicts_ps = self.extract_verbal_root_dicts()
        # Extract phrases based on verbal roots
        self.phrases_ps = self.extract_phrases()
        # Extract dictionaries mapping tokens to their deepest nominal heads within phrases
        self.nominal_root_dicts_pphr_ps = self.extract_nominal_root_dicts()
        # Extract noun phrases based on nominal roots
        self.noun_phrases_pphr_ps = self.extract_noun_phrases()

    def get_valid_sentences(self, conllu_file: str):
        """
        Filter the sentences obtained from the conllu file for the presence of verb tokens and consistency in
        head annotation.
        Return sentences and save original sentences to self.df_data.

        :param str conllu_file: Path to the conllu file
        :return: List of sentences where each sentence is a list of pyconll.unit.token.Token objects
        """
        # Load sentences from the CoNLL-U file
        conllu_sentences = pyconll.load_from_file(conllu_file)

        original_sentences_for_df = []
        sentences = []

        # Iterate through each sentence and filter out those without verbs or with invalid head annotations
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

            # Exclude tokens with IDs containing '-' (e.g., contractions)
            sentence = [word for word in sentence if '-' not in word.id]  # to exclude contractions present in some German datasets
            original_sentences_for_df.append(' '.join([word.form for word in sentence]))
            sentences.append(sentence)

        # Store original sentences in the dataframe
        self.df_data['original'] = original_sentences_for_df

        return sentences

    def extract_verbal_root_dicts(self):
        """
        For each token, extract information about it and find its deepest verbal head.
        Return dictionaries with token information saved as keys and the indices of the tokens' deepest verbal heads
        as values.

        :return: List of dictionaries containing information about every token as a key and the token's deepest verbal
        head as a value
        """
        path = self.file
        no_punc_lower_sentences_for_df = []
        verbal_root_dicts_ps = []

        # Iterate through each sentence to process tokens and find their deepest verbal roots
        for sentence in self.sentences:
            # Remove punctuation tokens from the sentence
            sent = [word for word in sentence if word.form not in string.punctuation]  # exclude punctuation here? what about the missing indices?
            no_punc_lower_sentences_for_df.append(' '.join(word.form.lower() for word in sent))
            verbal_root_dict = {}

            # For each token, recursively find its deepest verbal root
            for word in sent:
                word_id = int(word.id)
                word_cache = word.form.lower()  # .lower()
                word_deprel = word.deprel
                word_head = int(word.head)
                word_upos = word.upos
                word_feats = ''
                for key, value in word.feats.items():
                    word_feats = word_feats + f'{key}={list(value)[0]}|'
                word_feats = word_feats.rstrip('|')

                def find_root(word, path):
                    """
                    Recursively find the deepest verbal root for a given token within a sentence.
                    Verbs act as the central head of phrases, and this function ensures that each token is associated
                    with its deepest governing verb.

                    :param word: Token object containing dependency and part-of-speech information.
                    :param str path: Path to the conllu file
                    :return: None (updates verbal_root_dict in place)
                    """

                    def process_head(word):
                        """
                        Helper function to update verbal_root_dict and recursively call find_root.

                        :param word: Token object containing dependency and part-of-speech information
                        :return: None (updates verbal_root_dict in place)
                        """
                        head = sentence[int(word.head) - 1]
                        verbal_root_dict[(word_id, word_cache, word_deprel, word_head, word_upos, word_feats)] = head.head
                        find_root(head, path)

                    # Assign the current token as its own root initially
                    verbal_root_dict[(word_id, word_cache, word_deprel, word_head, word_upos, word_feats)] = word.id
                    # Use pattern matching to handle special cases for different treebanks/languages
                    match path:
                        case _ if any(substr in path for substr in {'da_ddt', 'nl_alpino', 'nl_lassysmall', 'fo_oft'}):
                            if int(word.head) != 0 and (word.upos not in {'VERB'} or {'Part'} == word.feats.get('VerbForm', None)):
                                process_head(word)
                        case _ if any(substr in path for substr in {'fo_farpahc'}):
                            if int(word.head) != 0 and (word.upos not in {'VERB'} or ({'Part'} == word.feats.get('VerbForm', None) and word.deprel == 'amod')):
                                process_head(word)
                        case _ if any(substr in path for substr in {'de_pud'}):
                            if int(word.head) != 0 and (word.upos not in {'VERB'} or word.deprel == 'compound:prt'):
                                process_head(word)
                        case _ if any(substr in path for substr in {'sv_talbanken'}):
                            if int(word.head) != 0 and (word.upos not in {'VERB'} or word.deprel == 'fixed'):
                                process_head(word)
                        case _:
                            if int(word.head) != 0 and word.upos not in {'VERB'}:
                                process_head(word)

                find_root(word, path)
            # Store the mapping for this sentence
            verbal_root_dicts_ps.append(verbal_root_dict)

        # Store lowercased, punctuation-free sentences in the dataframe
        self.df_data['no_punc_lower'] = no_punc_lower_sentences_for_df

        return verbal_root_dicts_ps

    def extract_phrases(self):
        """
        Group tokens into phrases based on their deepest verbal heads.
        Each phrase consists of tokens that share the same verbal root, with information about their dependencies.

        :return: List of lists, where each inner list contains tuples (min_token_idx, phrase_dict) in the order of their first token's index.
        """
        phrases_ps = []

        # For each sentence, group tokens by their verbal root to form phrases
        for vrd in self.verbal_root_dicts_ps:
            phrase_map = {}
            for key, value in vrd.items():
                if value not in phrase_map:
                    phrase_map[value] = []
                phrase_map[value].append(key)

            phrase_tuples = []
            for value, token_keys in phrase_map.items():
                min_idx = min(k[0] for k in token_keys)
                phrase_dict = {
                    (key[0], key[1]): {'deprel': key[2], 'head': key[3], 'upos': key[4], 'feats': key[5]}
                    for key in token_keys
                }
                phrase_tuples.append((min_idx, phrase_dict))
            phrase_tuples.sort(key=lambda x: x[0])
            phrases_ps.append(phrase_tuples)  # keep as list of (min_idx, phrase_dict) tuples

        return phrases_ps

    def extract_nominal_root_dicts(self):
        """
        Identify the deepest nominal head for each token within a phrase.
        Constructs dictionaries that map each token to its deepest nominal head within its phrase.

        :return: List of lists containing dictionaries. Each dictionary maps token information to its deepest nominal
        head within a phrase, with one list per sentence.
        """
        nominal_root_dicts_pphr_ps = []

        for sentence in self.phrases_ps:
            nominal_root_dicts_pphr = []
            for _, phr in sentence:
                nominal_root_dict = {}
                for word, features in phr.items():
                    word_id = word[0]
                    word_cache = word[1]
                    word_deprel = features['deprel']
                    word_head = features['head']
                    word_upos = features['upos']
                    word_feats = features['feats']

                    def find_nominal_root(word, features):
                        nominal_root_dict[(word_id, word_cache, word_deprel, word_head, word_upos, word_feats)] = word[0]
                        if features['head'] != 0 and features['upos'] not in {'NOUN'}:
                            try:
                                head, feats = [(h, f) for h, f in phr.items() if features['head'] == h[0]][0]
                                nominal_root_dict[(word_id, word_cache, word_deprel, word_head, word_upos, word_feats)] = feats['head']
                                find_nominal_root(head, feats)
                            except IndexError:
                                pass

                    find_nominal_root(word, features)

                nominal_root_dicts_pphr.append(nominal_root_dict)
            nominal_root_dicts_pphr_ps.append(nominal_root_dicts_pphr)

        return nominal_root_dicts_pphr_ps

    def extract_noun_phrases(self):
        """
        Group tokens into noun phrases based on their deepest nominal heads.
        Each noun phrase consists of tokens that share the same nominal root, with dependency information.

        :return: List of lists containing dictionaries. Each dictionary represents noun phrases in a sentence,
        where each phrase is identified by an index and contains tokens with their dependency information.
        """
        noun_phrases_pphr_ps = []

        for sentence in self.nominal_root_dicts_pphr_ps:
            noun_phrases_pphr = []
            for nrd in sentence:
                noun_phrase_dict = {}
                seen = set()
                for value in nrd.values():
                    if value not in seen:
                        noun_phrase_dict[len(seen)] = {
                            (key[0], key[1]): {'deprel': key[2], 'head': key[3], 'upos': key[4], 'feats': key[5]}
                            for key in nrd.keys() if nrd[key] == value
                        }
                        seen.add(value)
                noun_phrases_pphr.append(noun_phrase_dict)
            noun_phrases_pphr_ps.append(noun_phrases_pphr)

        return noun_phrases_pphr_ps

    # The Shuffler class is responsible for generating verb order error data by permuting verb positions within phrases
class Shuffler:
    """
    A class that processes and permutes verb phrases within sentences to generate verb order error data. The class
    extracts phrase structures, identifies valid and invalid token positions, permutes verb positions, generates labels,
    and stores the processed data in a dataframe.
    """
    def __init__(self, conllu_file: str, phrases: list[list[dict[int, dict]]], dataframe: pd.DataFrame):
        """
        Initialize the class with extracted phrases and a dataframe.
        Also generates various structured representations of the phrases for further processing.

        :param list[list[dict[int, dict]]]] phrases: List of lists containing phrase dictionaries for each sentence, where each phrase dictionary
        maps phrase indices to token information
        :param dataframe: Pandas DataFrame storing (un)permuted sentences and labels
        """
        self.file = conllu_file
        self.phrases_ps = phrases
        self.df_data = dataframe
        # Extract token positions and POS labels for all phrases
        self.all_id_pos_tok_phrases_ps, self.all_id_to_pos_tok_dicts_ps = self.get_id_pos_tok_phrases_and_dicts()
        # Generate position masks and correctness labels for all phrases
        self.positions_pphr_ps, self.inv_positions_pphr_ps, self.corr_l_pphr_ps = self.get_correct_gold_and_position_masks()
        # Create permuted sentences and corresponding error labels
        self.permuted_sents, self.incorrect_gold = self.create_verb_error_data()
        # Save the processed data to disk
        self.save_data()

    def get_id_pos_tok_phrases_and_dicts(self):
        """
        Extract token positions and their part-of-speech (POS) labels within phrases.
        Each phrase is represented as a structured mask indicating token identity and POS category.

        :return: Tuple containing:
             - List of lists, where each inner list represents a sentence's phrases as token-POS masks, ordered by min_idx.
             - List of dictionaries mapping token indices to their POS labels and token values.
        """
        all_id_pos_tok_phrases_ps = []
        all_id_to_pos_tok_dicts_ps = []

        for sentence in self.phrases_ps:
            id_pos_tok_phrases = []
            id_to_pos_tok = {}
            # Support both: list of (min_idx, phrase_dict) or just list of phrase_dict
            for phrase in sentence:
                if isinstance(phrase, tuple) and len(phrase) == 2:
                    # phrase is (min_idx, phrase_dict)
                    phrase_dict = phrase[1]
                else:
                    # phrase is just a phrase_dict (as in noun_phrases_pphr_ps)
                    phrase_dict = phrase
                phrase_mask = [0 for _ in range(len(phrase_dict.items()))]
                for idx, phr in phrase_dict.items():
                    # Mark tokens as 'V' (verb/aux) or 'O' (other) for each phrase
                    phr_mask = [(w[0], 'O', w[1]) if feats['upos'] not in {'VERB', 'AUX'} else (w[0], 'V', w[1]) for
                                w, feats in phr.items()]
                    phrase_mask[idx] = phr_mask
                    for w, feats in phr.items():
                        if feats['upos'] not in {'VERB', 'AUX'}:
                            id_to_pos_tok[w[0]] = ('O', w[1])
                        else:
                            id_to_pos_tok[w[0]] = ('V', w[1])
                id_pos_tok_phrases.append(phrase_mask)
            all_id_pos_tok_phrases_ps.append(id_pos_tok_phrases)
            all_id_to_pos_tok_dicts_ps.append(id_to_pos_tok)

        return all_id_pos_tok_phrases_ps, all_id_to_pos_tok_dicts_ps

    def get_correct_gold_and_position_masks(self):
        """
        Generate position masks and correctness labels for phrase tokens.
        Identifies valid and invalid token positions within phrases and assigns correctness labels based on the presence
        of verbs.

        :return: Tuple containing:
             - List of lists, where each inner list represents valid token positions within a phrase, ordered by min_idx.
             - List of lists, where each inner list represents invalid token positions within a phrase, ordered by min_idx.
             - List of lists, where each inner list represents correctness labels for token positions within a phrase, ordered by min_idx.
               Labels: 'C' (correct - verb token), 'O' (other tokens).
        """
        all_positions_pphr_ps = []
        all_invalid_positions_pphr_ps = []
        all_correct_labels_pphr_ps = []
        all_correct_labels_ps = []

        for i, id_pos_tok_phrases in enumerate(self.all_id_pos_tok_phrases_ps):
            all_positions_per_phrase = []
            all_invalid_positions_per_phrase = []
            correct_labels_per_phrase = []
            correct_labels_per_sentence = []

            for j, pos_tok_phrase in enumerate(id_pos_tok_phrases):
                all_positions = []
                all_invalid_positions = []
                for pos_tok in pos_tok_phrase:
                    for token in pos_tok:
                        all_positions.append(token[0])
                    if 'V' not in {token[1] for token in pos_tok}:
                        non_verb_phrase_positions = sorted([token[0] for token in pos_tok])
                        for k in range(len(non_verb_phrase_positions) - 1):
                            all_invalid_positions.append(non_verb_phrase_positions[k+1])
                all_positions_per_phrase.append(sorted(all_positions))
                all_invalid_positions_per_phrase.append(all_invalid_positions)

                position_copy = sorted(all_positions).copy()
                for pos_tok in id_pos_tok_phrases[j]:
                    for token in pos_tok:
                        idx = position_copy.index(token[0])
                        if token[1] == 'V':
                            position_copy[idx] = 'C'
                        else:
                            position_copy[idx] = 'O'
                correct_labels_per_phrase.append(position_copy)
                correct_labels_per_sentence.extend(position_copy)

            all_positions_pphr_ps.append(all_positions_per_phrase)
            all_invalid_positions_pphr_ps.append(all_invalid_positions_per_phrase)
            all_correct_labels_pphr_ps.append(correct_labels_per_phrase)
            all_correct_labels_ps.append(' '.join(correct_labels_per_sentence))

        # Store gold labels in the dataframe
        self.df_data['correct_gold'] = all_correct_labels_ps

        return all_positions_pphr_ps, all_invalid_positions_pphr_ps, all_correct_labels_pphr_ps

    def create_verb_error_data(self):
        """
        Generate data for verb placement errors by shuffling verb positions within phrases.
        Ensures that half of the time, verbs are placed in incorrect positions to create labeled verb order error data.

        For non-projective trees, reconstruct the sentence by placing each token (from all phrases) back into its original position,
        so that the linear order of the sentence is always preserved, but the verb tokens may be misplaced within their phrase boundaries.
        The labels are aligned with the permuted tokens.
        """
        all_permuted_sentences = []
        all_permuted_labels = []
        correct_counter = 0
        incorrect_counter = 0

        for i in range(len(self.positions_pphr_ps)):
            # For each sentence
            positions_per_phrase_per_sent = self.positions_pphr_ps[i]
            inv_positions_per_phrase_per_sent = self.inv_positions_pphr_ps[i]
            id_to_pos_tok_dict = self.all_id_to_pos_tok_dicts_ps[i]

            # Build a mapping from original token index to its permuted token and label
            token_idx_to_permuted_token = {}
            token_idx_to_permuted_label = {}

            # For each phrase in the sentence
            for j in range(len(positions_per_phrase_per_sent)):
                positions_pphr = positions_per_phrase_per_sent[j]
                inv_positions_pphr = inv_positions_per_phrase_per_sent[j]
                position_copy = positions_pphr.copy()
                possible_verb_positions = list(set(positions_pphr).difference(set(inv_positions_pphr)))
                last_position = position_copy[-1] + 1
                possible_verb_positions.append(last_position)

                # Shuffle verb positions within the phrase
                for id, pos_tok in id_to_pos_tok_dict.items():
                    trial_counter = 0
                    if pos_tok[0] == 'V' and id in positions_pphr:
                        verb_idx_within_phr = position_copy.index(id)

                        def find_new_idx(trial_counter):
                            nonlocal incorrect_counter, correct_counter
                            if incorrect_counter < correct_counter:
                                invalid_indices = [position_copy.index(inv) for inv in inv_positions_pphr]
                                if j == 0 and 0 not in invalid_indices:
                                    invalid_indices.insert(0, 0)
                                new_verb_position = random.choice(possible_verb_positions)
                                try:
                                    new_verb_idx_within_phr = position_copy.index(new_verb_position)
                                except ValueError:
                                    new_verb_idx_within_phr = new_verb_position
                                if new_verb_idx_within_phr not in invalid_indices:
                                    position_copy.insert(new_verb_idx_within_phr, id)
                                    if verb_idx_within_phr >= new_verb_idx_within_phr:
                                        position_copy.pop(verb_idx_within_phr + 1)
                                    else:
                                        position_copy.pop(verb_idx_within_phr)
                                elif trial_counter < math.factorial(len(possible_verb_positions)):
                                    trial_counter += 1
                                    find_new_idx(trial_counter)
                                else:
                                    pass
                            else:
                                pass

                        find_new_idx(trial_counter)

                # After shuffling, assign permuted tokens and labels to the new positions in the phrase
                # The new order of token indices in this phrase is position_copy
                # We want to place the permuted tokens into the original sentence order at the new positions

                # Build a list of (original_token_idx, permuted_token, label)
                phrase_permuted = []
                for idx_in_phrase, id in enumerate(position_copy):
                    orig_token_idx = id
                    token_str = id_to_pos_tok_dict[id][1]
                    # Determine label
                    orig_idx_in_phrase = positions_pphr.index(id)
                    if id_to_pos_tok_dict[id][0] != 'V':
                        label = 'O'
                    elif idx_in_phrase == orig_idx_in_phrase:
                        label = 'C'
                        correct_counter += 1
                    else:
                        label = 'F'
                        incorrect_counter += 1
                    phrase_permuted.append((idx_in_phrase, orig_token_idx, token_str, label))

                # Now, for each position in the phrase, assign the permuted token to the corresponding original token index
                # The phrase is being "shuffled" within itself, so we map the shuffled order back to the original token indices
                for idx_in_phrase, orig_token_idx, token_str, label in phrase_permuted:
                    # The idx_in_phrase-th position in the phrase now holds token_str, which should be placed at orig_token_idx in the sentence
                    token_idx_to_permuted_token[positions_pphr[idx_in_phrase]] = token_str
                    token_idx_to_permuted_label[positions_pphr[idx_in_phrase]] = label

            # Now, reconstruct the sentence in the original token order, but using the permuted tokens
            sorted_token_indices = sorted(token_idx_to_permuted_token.keys())
            permuted_sentence = ' '.join([token_idx_to_permuted_token[idx] for idx in sorted_token_indices])
            permuted_labels = ' '.join([token_idx_to_permuted_label[idx] for idx in sorted_token_indices])

            all_permuted_sentences.append(permuted_sentence)
            all_permuted_labels.append(permuted_labels)

        self.df_data['no_punc_lower_permuted'] = all_permuted_sentences
        self.df_data['permuted_gold'] = all_permuted_labels

        return all_permuted_sentences, all_permuted_labels

    def save_data(self):
        # TODO: Update this so it automatically splits no in nb and nn (for now, I manually created directories for holding these datasets)
        """
        Save the dataframe containing processed phrase data to a TSV file.

        :return: None
        """
        # Create output directory if it doesn't exist
        if 'verb_error_datasets_v1_1' not in os.listdir(os.getcwd()):
            os.mkdir('verb_error_datasets_v1_1')

        # Determine the data split (train/dev/test) based on the filename
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

        # Create language-specific output directory if it doesn't exist
        if identifier.split("_")[0] not in os.listdir('verb_error_datasets_v1_1'):
            os.mkdir(f'Datasets/verb_error_datasets_v1_1/{identifier.split("_")[0]}')

        # Save the dataframe as a TSV file in the appropriate directory
        self.df_data.to_csv(f'Datasets/verb_error_datasets_v1_1/{identifier.split("_")[0]}/{identifier}_{split}.tsv', sep='\t', encoding='utf-8', index_label='idx')

        # Confirmation message
        print(f"Finished processing and saving dataset: {identifier}_{split}")
