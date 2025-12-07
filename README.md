# GermDetect: Verb Placement Error Detection Datasets for Learners of Germanic Languages

This repository contains the code described in **Noah-Manuel Michael and Andrea Horbach (2025). GermDetect: Verb Placement Error Detection Datasets for Learners of Germanic Languages. In *Proceedings of the 20th Workshop on Building Educational Applications (BEA 2025)*. Association for Computational Linguistics.**

Specifically, you will find the code to create the GermDetect datasets and train prototype mBERT models for verb placement error detection. It also contains the datasets (ver. 1.0 and ver. 1.1).
In ver. 1.0, as explained in the Limitations section of the paper, we have identified a bug which — in sentences consisting of more than two verb-headed phrases — caused the order of the phrases to be restored incorrectly after introducing the verb placement errors. This bug has been fixed in ver. 1.1.
We reran the dataset creation algorithm and the training of the models; a summary of the updated results can be found in `/results/f05_scores_v1_1.txt` and below this paragraph.
The results stray only slightly from the original results reported in the paper, with differences in the range of **0.0001–0.0373 F0.5 points**. This implies that models seem to mostly rely on within-phrase information for the task. This intuitively makes sense, as the phrase type is often immediately inferrable from the inclusion of a conjunction, subjunction, or a relative pronoun (e.g., "weil" in German, "dat" in Dutch, "som" in Swedish, "fordi" in Danish) that explicitly marks the clause type, making long-range phrase order information less relevant for successfully classifying (in)correct verb placement within a clause. 
However, the Scandinavian languages, much like English, allow for the regular omission of such markers, e.g., Swedish "Jag går i dagar i tröjan (som) du hatar." ("I'll be wearing the sweater (that) you hate for days."). Nonetheless, this does not seem to cause the task to become significantly harder to solve.

![Updated results.](Experiments/results/results_v1_1.png)

# Technical Instructions

We do not include the UD datasets in this repository, as they are available under the CC BY 4.0 license and can be downloaded from the Universal Dependencies website.
A list of the UD datasets used to create the GermDetect datasets can be found in `/dataset_statistics/language_list.txt`.
In order to recreate the GermDetect datasets, you will need to download the UD datasets and place them in `/gerlangmod/Germanic UD/`.
The followings instructions assume a MacOS or Linux environment.

First, navigate to the root directory of this repository:
```
cd gerlangmod
```

Install the required Python packages in your environment by running the following command:
```
pip install -r requirements.txt
```

Then, to execute the main script that creates the GermDetect datasets and trains the mBERT models, run:
```
python3 main.py
```

This will create the GermDetect datasets in the 'Datasets/verb_error_datasets_v1_1/' directory.
It will also create the naive baselines datasets in the 'Datasets/verb_error_datasets_naive/' directory.

Finally, the script will train all configurations of the mBERT models on the GermDetect datasets described in the paper and save the resulting models in the 'models_v1_1' directory.

To get predictions on the test sets, run:
```
python3 inference_v1_1.py
```

To calculate the F0.5 scores of the predictions, run:
```
python3 eval.py
```

This will write the F0.5 scores to the file `Experiments/results/f05_scores_v1_1.txt`.

# Verb Order Error Dataset Creation Algorithm

## Overview
GermDetect employs an algorithm that introduces errors in verb placement within well-formed sentences from Universal Dependencies (UD) datasets. 
The core assumption is that these sentences are grammatically correct, and their UPOS and dependency annotations are accurate. The algorithm 
randomly reorders verb tokens within their respective phrases while leaving noun phrases intact, i.e., no verb can be placed within elements of a single noun phrase.

## Method
For every sentence containing at least one verb token, the algorithm performs the following steps:

1. **Phrase Extraction**: Identifies and aggregates full verb tokens and their dependent tokens into phrases.
2. **Noun Phrase Identification**: Within these phrases, noun-headed groups of tokens are treated as impermeable units to prevent disruption by verb reordering.
3. **Verb Reordering**: Randomly selects and repositions verb tokens within their respective phrases while keeping the noun phrases intact.
4. **Sentence-Initial Constraints**: If the manipulated phrase is the first in a sentence, verbs are restricted from occupying the initial position to avoid accidentally forming correct polar question syntax in Germanic languages.

## Example
Consider the following input sentence (common punctuation is removed, and tokens are lowercased to eliminate contextual clues beyond word order):

**Original Sentence:** Dutch for "I know that he has bought a dog."
```
ik weet dat hij een hond heeft gekocht
```

**Step 1: Extraction of Verb-Headed Phrases**
```
[ik weet]
[dat hij een hond heeft gekocht]
```

**Step 2: Extraction of Noun-Headed Phrases and Impermeabilization Thereof**
```
[ik weet]
[dat hij (een hond) heeft gekocht]
```

**Step 3: Verb Reordering**
```
[ik weet]
[dat gekocht hij heeft (een hond)]
```

**Analysis of Changes:**
1. *1st phrase:* `weet` stays in its original position because the only possible permutation of the phrase would result in correct polar question syntax (`weet ik`).
2. *2nd phrase:* `heeft` and `gekocht` are misplaced.

**Output Labeling**

The algorithm labels tokens as follows:
- `O` (Other) - Non-verb tokens
- `C` (Correct) - Correctly placed verb tokens
- `F` (False) - Incorrectly placed verb tokens

```
ik  weet    dat gekocht hij heeft   een hond
O   C       O   F       O   F       O   O
```

## Label Distribution
The algorithm ensures that the distribution of correctly (`C`) and incorrectly (`F`) placed verbs in each dataset remains approximately equal.

