# Noah-Manuel Michael
# Created: 2025-02-18
# Updated: 2025-07-01
# Script to apply the finetuned LLM on unseen verb error data

import os
import pandas as pd
import torch
import shutil
from transformers import BertForTokenClassification, BertTokenizerFast


# Check and prepare predictions_v1_1 directory
PRED_DIR = "predictions_v1_1"
SRC_DIR = "../Datasets/verb_error_datasets_v1_1"

if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)
    # Copy all subdirectories from verb_error_datasets_v1_1 to predictions_v1_1
    for subdir in os.listdir(SRC_DIR):
        src_path = os.path.join(SRC_DIR, subdir)
        dst_path = os.path.join(PRED_DIR, subdir)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
else:
    # Remove all contents of predictions_v1_1 for a fresh start
    for entry in os.listdir(PRED_DIR):
        entry_path = os.path.join(PRED_DIR, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)
    # Copy all subdirectories from verb_error_datasets_v1_1 to predictions_v1_1
    for subdir in os.listdir(SRC_DIR):
        src_path = os.path.join(SRC_DIR, subdir)
        dst_path = os.path.join(PRED_DIR, subdir)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)

label2id = {"O": 0, "F": 1, "C": 2}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping

# Device (MPS on Mac or CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load model from the latest checkpoint
for language in os.listdir('models_v1_1'):
    if language != 'all_langs':
        for configuration in os.listdir(f'Experiments/models_v1_1/{language}'):
            for checkpoint in os.listdir(f'Experiments/models_v1_1/{language}/{configuration}'):

                model = BertForTokenClassification.from_pretrained(
                    f"Experiments/models_v1_1/{language}/{configuration}/{checkpoint}")
                tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
                max_len = tokenizer.model_max_length  # usually 512 for BERT

                # Sample text
                for dataset in os.listdir(f'predictions_v1_1/{language}'):
                    if 'test' in dataset:
                        test_df = pd.read_csv(f"predictions_v1_1/{language}/{dataset}", sep="\t")
                        texts = test_df['no_punc_lower_permuted'].tolist()

                        all_labels = []

                        for text in texts:
                            words = text.split()
                            chunk_size = max_len - 2  # leave room for [CLS]/[SEP]
                            chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]

                            sentence_labels = []
                            for chunk in chunks:
                                encoding = tokenizer(
                                    chunk,
                                    is_split_into_words=True,
                                    return_tensors="pt",
                                    padding=False,
                                    truncation=False,
                                )
                                with torch.no_grad():
                                    logits = model(**encoding).logits
                                preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
                                word_ids = encoding.word_ids(batch_index=0)

                                prev_word_idx = None
                                for idx, word_idx in enumerate(word_ids):
                                    if word_idx is None or word_idx == prev_word_idx:
                                        continue
                                    sentence_labels.append(id2label[preds[idx]])
                                    prev_word_idx = word_idx

                            # pad or trim in the unlikely event of mismatch
                            if len(sentence_labels) != len(words):
                                print(f"Alignment mismatch in '{text[:50]}…': {len(words)} words vs {len(sentence_labels)} labels")
                                if len(sentence_labels) > len(words):
                                    sentence_labels = sentence_labels[: len(words)]
                                else:
                                    sentence_labels += ["O"] * (len(words) - len(sentence_labels))

                            all_labels.append(' '.join(sentence_labels))

                        test_df[configuration] = all_labels
                        test_df.to_csv(f"predictions_v1_1/{language}/{dataset}", index=False, sep="\t")
                        print(f"Wrote predictions for {language}/{dataset}: config={configuration}")

# Tokenizer (shared across all models)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
max_len = tokenizer.model_max_length  # usually 512 for BERT

# Directory where the five “all_langs” models live
ALL_LANGS_DIR = "models_v1_1/all_langs"

# Decide which configs to run for each language
def get_configs_for_language(lang):
    if lang in ("af", "de", "nl"):
        return ["all", "west"]
    if lang in ("fo", "is"):
        return ["all", "island", "north"]
    if lang in ("da", "nb", "nn", "sv"):
        return ["all", "mainland", "north"]
    return []

for language in os.listdir("models_v1_1"):
    if language == "all_langs":
        continue

    configs = get_configs_for_language(language)
    if not configs:
        continue  # skip anything outside the specified sets

    # iterate over each test file for this language
    data_dir = f"predictions_v1_1/{language}"
    for dataset in os.listdir(data_dir):
        if "test" not in dataset:
            continue

        df_path = os.path.join(data_dir, dataset)
        test_df = pd.read_csv(df_path, sep="\t")
        texts = test_df["no_punc_lower_permuted"].tolist()

        # for each configuration, load its model(s) and predict
        for config in configs:
            config_dir = os.path.join(ALL_LANGS_DIR, config)

            # if there are multiple checkpoints under each config, loop them;
            # if there’s only one, this still works
            for checkpoint in os.listdir(config_dir):
                model_path = os.path.join(config_dir, checkpoint)
                model = BertForTokenClassification.from_pretrained(
                    model_path, num_labels=len(label2id)
                ).to(device)
                model.eval()

                all_labels = []
                for text in texts:
                    words = text.split()
                    chunk_size = max_len - 2  # leave room for [CLS]/[SEP]
                    chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]

                    sentence_labels = []
                    for chunk in chunks:
                        encoding = tokenizer(
                            chunk,
                            is_split_into_words=True,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                        ).to(device)

                        with torch.no_grad():
                            logits = model(**encoding).logits
                        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
                        word_ids = encoding.word_ids(batch_index=0)

                        prev_word_idx = None
                        for idx, word_idx in enumerate(word_ids):
                            if word_idx is None or word_idx == prev_word_idx:
                                continue
                            sentence_labels.append(id2label[preds[idx]])
                            prev_word_idx = word_idx

                    # pad or trim in the unlikely event of mismatch
                    if len(sentence_labels) != len(words):
                        print(f"Alignment mismatch in '{text[:50]}…': {len(words)} words vs {len(sentence_labels)} labels")
                        if len(sentence_labels) > len(words):
                            sentence_labels = sentence_labels[: len(words)]
                        else:
                            sentence_labels += ["O"] * (len(words) - len(sentence_labels))

                    all_labels.append(" ".join(sentence_labels))

                # write the column named after the config
                test_df[config] = all_labels

        # save all predictions for this test file
        test_df.to_csv(df_path, sep="\t", index=False)
        print(f"Wrote predictions for {language}/{dataset}: configs={configs}")
