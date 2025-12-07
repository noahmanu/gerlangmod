# Noah-Manuel Michael
# Created: 2025-02-18
# Updated: 2025-06-29
# Script to finetune LLMs on verb error data

import transformers
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict


# Set random seed for reproducibility
transformers.set_seed(667)


def train_model(df_train, df_dev, configuration, language):
    """
    Fine-tune a BERT-based token classification model on verb error data.

    :param pandas.DataFrame df_train:
        The training data containing sentences and corresponding token-level labels.
    :param pandas.DataFrame df_dev:
        The development/validation data containing sentences and corresponding token-level labels.
    :param str configuration:
        Specifies the data configuration/strategy (e.g., 'target', 'all', 'random', etc.).
    :param str language:
        The target language code (e.g., 'de', 'nl', etc.).
    :return: None
        The function trains and saves the best model to disk.
    """
    # Select the appropriate columns for sentences and labels based on configuration
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

    # Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    # Define label mappings
    label2id = {"O": 0, "F": 1, "C": 2}
    # id2label = {v: k for k, v in label2id.items()}  # Reverse mapping
    num_labels = len(label2id)

    # Tokenization and label alignment function
    def tokenize_and_align_labels(examples):
        all_tokenized_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []}

        # Tokenize each sentence and align the labels to the wordpieces
        for sentence, labels in zip(examples[sentence_column], examples[label_column]):
            tokenized_inputs = tokenizer(
                sentence.split(),
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True)

            word_ids = tokenized_inputs.word_ids()
            label_list = labels.split()

            encoded_labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    encoded_labels.append(-100)
                elif word_id != previous_word_id:
                    if word_id < len(label_list):
                        encoded_labels.append(label2id[label_list[word_id]])
                    else:
                        # If label is missing or too short: fallback to -100
                        encoded_labels.append(-100)
                else:
                    encoded_labels.append(-100)
                previous_word_id = word_id

            all_tokenized_inputs["input_ids"].append(tokenized_inputs["input_ids"])
            all_tokenized_inputs["attention_mask"].append(tokenized_inputs["attention_mask"])
            all_tokenized_inputs["labels"].append(encoded_labels)

        return all_tokenized_inputs

    # Convert pandas DataFrames to HuggingFace Datasets
    datasets = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "validation": Dataset.from_pandas(df_dev)})

    # Apply tokenization and label alignment to the datasets
    datasets = datasets.map(tokenize_and_align_labels, batched=True)

    # Load the model for token classification
    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./models_v1_1/{language}/{configuration}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs/{language}/{configuration}",
        report_to=["tensorboard"],
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer)

    # Start training
    trainer.train()

    # Confirmation message after training
    print(f"Finished training model for language '{language}' with configuration '{configuration}' and saved to disk.")
