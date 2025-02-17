# Noah-Manuel Michael
# Created: 2025-01-09
# Updated: 2025-02-17
# Classes for creating verb error data

import os
import string
import pyconll
import pandas as pd
import random
import math
import re
# import pprint

# pp = pprint.PrettyPrinter(indent=0)
# pd.set_option("display.max_columns", None)


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
        :param str conllu_file: The path to the conllu file to be processed
        """
        self.file = conllu_file
        self.df_data = pd.DataFrame()
        self.sentences = self.get_valid_sentences(self.file)
        self.verbal_root_dicts_ps = self.extract_verbal_root_dicts()
        self.phrases_ps = self.extract_phrases()
        self.nominal_root_dicts_pphr_ps = self.extract_nominal_root_dicts()
        self.noun_phrases_pphr_ps = self.extract_noun_phrases()

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

            sentence = [word for word in sentence if '-' not in word.id]  # to exclude contractions present in some German datasets
            original_sentences_for_df.append(' '.join([word.form for word in sentence]))
            sentences.append(sentence)

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

        for sentence in self.sentences:
            sent = [word for word in sentence if word.form not in string.punctuation]  # exclude punctuation here? what about the missing indices?
            no_punc_lower_sentences_for_df.append(' '.join(word.form.lower() for word in sent))
            verbal_root_dict = {}

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

                    verbal_root_dict[(word_id, word_cache, word_deprel, word_head, word_upos, word_feats)] = word.id
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
            verbal_root_dicts_ps.append(verbal_root_dict)

        self.df_data['no_punc_lower'] = no_punc_lower_sentences_for_df

        return verbal_root_dicts_ps

    def extract_phrases(self):
        """
        Group tokens into phrases based on their deepest verbal heads.
        Each phrase consists of tokens that share the same verbal root, with information about their dependencies.

        :return: List of dictionaries where each dictionary represents phrases in a sentence and each phrase is
        identified by an index and contains tokens with their dependency information
        """
        phrases_ps = []

        for vrd in self.verbal_root_dicts_ps:
            phrase_dict = {}
            seen = set()

            for value in vrd.values():
                if value not in seen:
                    phrase_dict[len(seen)] = {
                        (key[0], key[1]): {'deprel': key[2], 'head': key[3], 'upos': key[4], 'feats': key[5]}
                        for key in vrd.keys() if vrd[key] == value
                    }
                    seen.add(value)
            phrases_ps.append(phrase_dict)

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

            for i, phr in sentence.items():
                nominal_root_dict = {}

                for word, features in phr.items():
                    word_id = word[0]
                    word_cache = word[1]
                    word_deprel = features['deprel']
                    word_head = features['head']
                    word_upos = features['upos']
                    word_feats = features['feats']

                    def find_nominal_root(word, features):
                        """
                        Recursively find the deepest nominal root for a given token within a phrase.
                        Updates the nominal root dictionary with the identified root.

                        :param word: Tuple representing the token (word index, word form).
                        :param features: Dictionary containing dependency and part-of-speech information for the token.
                        :return: None
                        """
                        nominal_root_dict[(word_id, word_cache, word_deprel, word_head, word_upos, word_feats)] = word[0]
                        if features['head'] != 0 and features['upos'] not in {'NOUN'}:
                            try:
                                head, feats = [(h, f) for h, f in phr.items() if features['head'] == h[0]][0]
                                nominal_root_dict[(word_id, word_cache, word_deprel, word_head, word_upos, word_feats)] = feats['head']  # check here if correct features or feats?
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


class Shuffler:
    """
    A class that processes and permutes verb phrases within sentences to generate verb order error data. The class
    extracts phrase structures, identifies valid and invalid token positions, permutes verb positions, generates labels,
    and stores the processed data in a dataframe.
    """
    def __init__(self, conllu_file: str, phrases: list[list[dict[int, dict[tuple, dict]]]], dataframe: pd.DataFrame):
        """
        Initialize the class with extracted phrases and a dataframe.
        Also generates various structured representations of the phrases for further processing.

        :param list[list[dict[int, dict[tuple, dict]]]] phrases: List of lists containing phrase dictionaries for each sentence, where each phrase dictionary
        maps phrase indices to token information
        :param dataframe: Pandas DataFrame storing (un)permuted sentences and labels
        """
        self.file = conllu_file
        self.phrases_ps = phrases
        self.df_data = dataframe
        self.all_id_pos_tok_phrases_ps, self.all_id_to_pos_tok_dicts_ps = self.get_id_pos_tok_phrases_and_dicts()
        self.positions_pphr_ps, self.inv_positions_pphr_ps, self.corr_l_pphr_ps = self.get_correct_gold_and_position_masks()
        self.permuted_sents, self.incorrect_gold = self.create_verb_error_data()
        self.save_data()

    def get_id_pos_tok_phrases_and_dicts(self):
        """
        Extract token positions and their part-of-speech (POS) labels within phrases.
        Each phrase is represented as a structured mask indicating token identity and POS category.

        :return: Tuple containing:
             - List of lists, where each inner list represents a sentence's phrases as token-POS masks.
             - List of dictionaries mapping token indices to their POS labels and token values.
        """
        all_id_pos_tok_phrases_ps = []
        all_id_to_pos_tok_dicts_ps = []

        for sentence in self.phrases_ps:
            id_pos_tok_phrases = []
            id_to_pos_tok = {}
            for phrase in sentence:
                phrase_mask = [0 for _ in range(len(phrase.items()))]
                for idx, phr in phrase.items():
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
             - List of lists, where each inner list represents valid token positions within a phrase.
             - List of lists, where each inner list represents invalid token positions within a phrase.
             - List of lists, where each inner list represents correctness labels for token positions within a phrase.
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

        self.df_data['correct_gold'] = all_correct_labels_ps

        return all_positions_pphr_ps, all_invalid_positions_pphr_ps, all_correct_labels_pphr_ps

    def create_verb_error_data(self):
        """
        Generate data for verb placement errors by shuffling verb positions within phrases.
        Ensures that half ot the time, verbs are placed in incorrect positions to create labeled verb order error data.

        :return: Tuple containing:
             - List of sentences with permuted verb positions, where each phrase is represented
               as a string of tokens.
             - List of corresponding correctness labels for each token in the permuted sentences.
               Labels: 'C' (correct verb position), 'F' (incorrect verb position), 'O' (other tokens).
        """
        all_filled_phrases = []
        all_labels = []
        correct_counter = 0
        incorrect_counter = 0

        for i, positions_per_phrase_per_sent in enumerate(self.positions_pphr_ps):
            filled_phr_ps = []
            labels_ps = []

            for j, positions_pphr in enumerate(positions_per_phrase_per_sent):
                position_copy = positions_pphr.copy()
                possible_verb_positions = list(set(positions_pphr).difference(set(self.inv_positions_pphr_ps[i][j])))
                last_position = position_copy[-1] + 1  # create a last dummy position
                possible_verb_positions.extend([last_position])  # add a dummy position to make it possible for the verb to be placed behind all other tokens, otherwise with the insert method only second-to-last would be possible

                for id, pos_tok in self.all_id_to_pos_tok_dicts_ps[i].items():
                    trial_counter = 0
                    if pos_tok[0] == 'V' and id in positions_pphr:
                        verb_idx_within_phr = position_copy.index(id)

                        def find_new_idx(trial_counter):
                            """
                            Find a new position for a verb within the phrase, ensuring that verbs are placed
                            incorrectly half of the time.

                            :param trial_counter: Counter tracking the number of trials to find a new position.
                            :return: None
                            """
                            if incorrect_counter < correct_counter:
                                invalid_indices = [position_copy.index(inv) for inv in self.inv_positions_pphr_ps[i][j]]
                                if j == 0 and 0 not in invalid_indices:  # if first phrase in sentence, no verb can be in the first position because it would unintentionally create polar question syntax
                                    invalid_indices.insert(0, 0)
                                new_verb_position = random.choice(possible_verb_positions)
                                try:
                                    new_verb_idx_within_phr = position_copy.index(new_verb_position)
                                except ValueError:
                                    new_verb_idx_within_phr = new_verb_position  # if it picks the dummy position, it will be out of range, so we can use the dummy position as an idx for insertion directly, since OOR indices get inserted at the last position
                                if new_verb_idx_within_phr not in invalid_indices:  # this means that if the last dummy position is picked, it will always be okay to put the verb there, do I want this? Or exclude the last token if a verb from taking the dummy position?
                                    position_copy.insert(new_verb_idx_within_phr, id)
                                    if verb_idx_within_phr >= new_verb_idx_within_phr:
                                        position_copy.pop(verb_idx_within_phr + 1)
                                    else:
                                        position_copy.pop(verb_idx_within_phr)
                                elif trial_counter < math.factorial(len(possible_verb_positions)):
                                    trial_counter += 1
                                    find_new_idx(trial_counter)
                                else:
                                    pass  # if no suitable new position for the verb can be found, pass
                            else:
                                pass

                        find_new_idx(trial_counter)

                for inv_position in self.inv_positions_pphr_ps[i][j]:  # for invalid position
                    if inv_position - 1 in positions_pphr and self.all_id_to_pos_tok_dicts_ps[i][inv_position - 1][0] != 'V':  # if the element before it exists (i.e., was not a punctuation token that got filtered out) and is not a verb token itself
                        assert position_copy[position_copy.index(inv_position) - 1] == inv_position - 1, 'An invalid swap has taken place.'  # check that the invalid position follows the element that is supposed to come before it directly
                for position in positions_pphr:
                    assert position in position_copy, 'All positions aren\'t present in the shuffled phrase.'

                filled = position_copy.copy()
                for id, pos_tok in self.all_id_to_pos_tok_dicts_ps[i].items():
                    if id in positions_pphr:
                        filled[position_copy.index(id)] = pos_tok[1]

                labels = position_copy.copy()
                for id, pos_tok in self.all_id_to_pos_tok_dicts_ps[i].items():
                    if id in positions_pphr:
                        if pos_tok[0] != 'V':
                            labels[position_copy.index(id)] = 'O'
                        elif pos_tok[0] == 'V' and position_copy.index(id) == positions_pphr.index(id):
                            labels[position_copy.index(id)] = 'C'
                            correct_counter += 1
                        else:
                            labels[position_copy.index(id)] = 'F'
                            incorrect_counter += 1

                filled_phr_ps.append(' '.join(filled))
                labels_ps.append(' '.join(labels))

            all_filled_phrases.append(' '.join(filled_phr_ps))
            all_labels.append(' '.join(labels_ps))

        self.df_data['no_punc_lower_permuted'] = all_filled_phrases
        self.df_data['permuted_gold'] = all_labels

        return all_filled_phrases, all_labels

    def save_data(self):
        """
        Save the dataframe containing processed phrase data to a TSV file.

        :return: None
        """
        if 'verb_error_datasets' not in os.listdir(os.getcwd()):
            os.mkdir('verb_error_datasets')

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

        if identifier.split("_")[0] not in os.listdir('verb_error_datasets'):
            os.mkdir(f'verb_error_datasets/{identifier.split("_")[0]}')

        self.df_data.to_csv(f'verb_error_datasets/{identifier.split("_")[0]}/{identifier}_{split}.tsv', sep='\t', encoding='utf-8', index_label='idx')
