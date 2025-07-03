# Noah-Manuel Michael
# Created: 2025-04-21
# Updated: 2025-06-29
# Script to compute the F05 scores on the predicted data

import os
import pandas as pd
from sklearn.metrics import fbeta_score

label2id = {"O": 0, "F": 1, "C": 2}

def compute_f05_score(gold, pred):
    return fbeta_score(
        [label2id[l] for l in gold],
        [label2id[l] for l in pred],
        beta=0.5,
        average="macro",
        labels=[1, 2]  # Only take "F" and "C" labels into account for evaluation
    )

# Define per‑language configs
language_config_map = {
    "af": ["target", "all", "all-balanced", "west", "west-balanced", "random", "adjacent"],
    "de": ["target", "all", "all-balanced", "west", "west-balanced", "random", "adjacent"],
    "nl": ["target", "all", "all-balanced", "west", "west-balanced", "random", "adjacent"],
    "fo": ["target", "all", "all-balanced", "north", "north-balanced", "island", "island-balanced", "random", "adjacent"],
    "is": ["target", "all", "all-balanced", "north", "north-balanced", "island", "island-balanced", "random", "adjacent"],
    "da": ["target", "all", "all-balanced", "north", "north-balanced", "mainland", "mainland-balanced", "random", "adjacent"],
    "nb": ["target", "all", "all-balanced", "north", "north-balanced", "mainland", "mainland-balanced", "random", "adjacent"],
    "nn": ["target", "all", "all-balanced", "north", "north-balanced", "mainland", "mainland-balanced", "random", "adjacent"],
    "sv": ["target", "all", "all-balanced", "north", "north-balanced", "mainland", "mainland-balanced", "random", "adjacent"],
}

root_dir = "predictions_v1_1"
results = {}

for language, configs_to_evaluate in language_config_map.items():
    gold_labels = []
    config_preds = {cfg: [] for cfg in configs_to_evaluate}

    lang_dir = os.path.join(root_dir, language)
    if not os.path.isdir(lang_dir):
        continue

    for fname in os.listdir(lang_dir):
        if "test" not in fname:
            continue

        df_path = os.path.join(lang_dir, fname)
        df = pd.read_csv(df_path, sep="\t")

        # Collect gold labels once per sentence
        if "permuted_gold" not in df.columns:
            print(f"No 'permuted_gold' column in {language}/{fname}")
            continue

        gold_seqs = df["permuted_gold"].str.split().tolist()
        for gold_sent in gold_seqs:
            gold_labels.extend(gold_sent)

        # Collect predictions_v1_0 per config
        for cfg in configs_to_evaluate:
            if cfg not in df.columns:
                continue

            # Due to a tokenization mismatch in the Swedish dataset, where a couple of sentences use Chinese characters, we need to ensure that the lengths match
            pred_seqs = df[cfg].str.split().tolist()
            for gold_sent, pred_sent in zip(gold_seqs, pred_seqs):
                if len(gold_sent) != len(pred_sent):
                    # Pad or truncate to match length
                    if len(pred_sent) > len(gold_sent):
                        pred_sent = pred_sent[:len(gold_sent)]
                    else:
                        pred_sent += ["O"] * (len(gold_sent) - len(pred_sent))
                config_preds[cfg].extend(pred_sent)

    # Compute F0.5 for each config
    results[language] = {}
    for cfg in configs_to_evaluate:
        preds = config_preds[cfg]
        if len(preds) == len(gold_labels) and preds:
            score = compute_f05_score(gold_labels, preds)
            results[language][cfg] = round(score, 4)
        else:
            # Trim or pad globally if still mismatched
            min_len = min(len(gold_labels), len(preds))
            trimmed_gold = gold_labels[:min_len]
            trimmed_pred = preds[:min_len] + ["O"] * max(0, min_len - len(preds))
            score = compute_f05_score(trimmed_gold, trimmed_pred)
            results[language][cfg] = round(score, 4)
            print(f"Global length mismatch for {language}/{cfg}: "
                  f"gold={len(gold_labels)} pred={len(preds)} → trimmed to {min_len}")

# Write to text file
with open("results/f05_scores_v1_1.txt", "w", encoding="utf-8") as f:
    for lang, scores in results.items():
        f.write(f"{lang}\n")
        for cfg, sc in scores.items():
            f.write(f"  {cfg:20s} F0.5 = {sc:.4f}\n")
        f.write("\n")

print(f"F0.5 scores saved to f05_scores_v1_1.txt")



