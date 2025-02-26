# Noah-Manuel Michael
# Created: 2025-02-18
# Updated: 2025-02-18
# Script to finetune LLMs on verb error data

import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load data
df_train = pd.read_csv('verb_error_datasets/is/is_gc_train.tsv', sep='\t', encoding='utf-8', index_col=0)
df_dev = pd.read_csv('verb_error_datasets/is/is_gc_dev.tsv', sep='\t', encoding='utf-8', index_col=0)
df_test = pd.read_csv('verb_error_datasets/is/is_gc_test.tsv', sep='\t', encoding='utf-8', index_col=0)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")


# Define a fixed label-to-index mapping
label2id = {"O": 0, "F": 1, "C": 2}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping
num_labels = len(label2id)


def tokenize_and_align_labels(examples):
    all_tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for text, labels in zip(examples["no_punc_lower_permuted"], examples["permuted_gold"]):
        # Tokenize the input sentence (words must be split)
        tokenized_inputs = tokenizer(
            text.split(),  # Split to ensure token-level alignment
            truncation=True,
            padding="max_length",
            max_length=128,
            is_split_into_words=True
        )

        word_ids = tokenized_inputs.word_ids()  # Get word mappings
        label_list = labels.split()  # Convert label string into a list

        # Ensure label list and word_ids align properly
        encoded_labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:  # Special tokens (CLS, SEP, PAD)
                encoded_labels.append(-100)
            elif word_id != previous_word_id:  # First subword of the word
                encoded_labels.append(label2id[label_list[word_id]])
            else:  # Subword gets ignored label
                encoded_labels.append(-100)
            previous_word_id = word_id

        # Store results in batch-friendly format
        all_tokenized_inputs["input_ids"].append(tokenized_inputs["input_ids"])
        all_tokenized_inputs["attention_mask"].append(tokenized_inputs["attention_mask"])
        all_tokenized_inputs["labels"].append(encoded_labels)

    return all_tokenized_inputs


# Convert pandas DataFrame to Hugging Face Dataset
def convert_to_hf_dataset(df):
    return Dataset.from_pandas(df)


datasets = DatasetDict({
    "train": convert_to_hf_dataset(df_train),
    "validation": convert_to_hf_dataset(df_dev),
    "test": convert_to_hf_dataset(df_test),
})

datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Load model
model = BertForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=num_labels
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate
trainer.evaluate(datasets["test"])

