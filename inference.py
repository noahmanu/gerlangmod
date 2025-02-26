# Noah-Manuel Michael
# Created: 2025-02-18
# Updated: 2025-02-18
# Script to apply the finetuned LLM on unseen verb error data

from transformers import BertForTokenClassification, BertTokenizerFast, pipeline

label2id = {"O": 0, "F": 1, "C": 2}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping

# Load model from the latest checkpoint
model = BertForTokenClassification.from_pretrained("./results/checkpoint-1467")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

# Set up pipeline using the loaded model and tokenizer
token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Sample text
text = "þar er hægt að sér dagskrá ráðstefnunnar í dag og á morgun kynna"

output = token_classifier(text)
# Extract only the labels from the output
labels = []
previous_word = None

for item in output:
    word = item["word"]
    if not word.startswith("##"):  # Ignore subword labels
        label_index = int(item["entity"].split("_")[-1])  # Extract numeric part from LABEL_x
        labels.append(id2label[label_index])  # Map back to O, F, C

print(labels)



