# Noah-Manuel Michael
# Created: 2025-02-18
# Updated: 2025-06-29
# Script to finetune LLMs on verb error data

import random
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

random.seed(667)


def train_model(df_train, df_dev, configuration, language):
    """

    :param df_train:
    :param df_dev:
    :param configuration:
    :param language:
    :return:
    """
    if configuration in ['target', 'all', 'all-balanced', 'west', 'west-balanced', 'north', 'north-balanced', 'island',
                         'island-balanced', 'mainland', 'mainland-balanced']:
        sentence_column = 'no_punc_lower_permuted'
        label_column = 'permuted_gold'
    elif configuration in ['random']:
        sentence_column = 'random'
        label_column = 'random_gold'
    elif configuration in ['adjacent']:
        sentence_column = 'adjacent'
        label_column = 'adjacent_gold'

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    label2id = {"O": 0, "F": 1, "C": 2}
    # id2label = {v: k for k, v in label2id.items()}  # Reverse mapping
    num_labels = len(label2id)

    def tokenize_and_align_labels(examples):
        all_tokenized_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for sentence, labels in zip(examples[sentence_column], examples[label_column]):
            tokenized_inputs = tokenizer(
                sentence.split(),
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True
            )

            word_ids = tokenized_inputs.word_ids()
            label_list = labels.split()

            encoded_labels = []
            previous_word_id = None
            # for word_id in word_ids:
            #     if word_id is None:  # Special tokens (CLS, SEP, PAD)
            #         encoded_labels.append(-100)
            #     elif word_id != previous_word_id:  # First subword of the word
            #         encoded_labels.append(label2id[label_list[word_id]])
            #     else:  # Subword gets ignored label
            #         encoded_labels.append(-100)
            #     previous_word_id = word_id
            for word_id in word_ids:
                if word_id is None:
                    encoded_labels.append(-100)
                elif word_id != previous_word_id:
                    if word_id < len(label_list):
                        encoded_labels.append(label2id[label_list[word_id]])
                    else:
                        # Falls das Label fehlt oder zu kurz ist: Fallback zu -100
                        encoded_labels.append(-100)
                else:
                    encoded_labels.append(-100)
                previous_word_id = word_id

            all_tokenized_inputs["input_ids"].append(tokenized_inputs["input_ids"])
            all_tokenized_inputs["attention_mask"].append(tokenized_inputs["attention_mask"])
            all_tokenized_inputs["labels"].append(encoded_labels)

        return all_tokenized_inputs

    datasets = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "validation": Dataset.from_pandas(df_dev),
    })

    datasets = datasets.map(tokenize_and_align_labels, batched=True)

    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=f"./models_v1_0/{language}/{configuration}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs/{language}/{configuration}",
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()


# languages_per_conf = {
#     'target': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
#     'random': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
#     'adjacent': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
#     'all': ['all_lang'],
#     'all-balanced': ['af', 'da', 'de', 'fo', 'is', 'nl', 'nb', 'nn', 'sv'],
#     'west': ['all_langs'],
#     'west-balanced': ['af', 'de', 'nl'],
#     'north': ['all_langs'],
#     'north-balanced': ['da', 'fo', 'is', 'nb', 'nn', 'sv'],
#     'island': ['all_langs'],
#     'island-balanced': ['fo', 'is'],
#     'mainland': ['all_langs'],
#     'mainland-balanced': ['da', 'nb', 'nn', 'sv'],
# }
#
# for config, langs in languages_per_conf.items():
#     for lang in langs:
#         train_df, dev_df = load_relevant_train_datasets(config, lang)
