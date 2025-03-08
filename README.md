# gerlangmod
Code repository for the Germanic Language Modeling paper to be written for TrustLLM WP7.

# Verb Order Error Dataset Creation Algorithm

## Overview
This project implements an algorithm that introduces errors in verb placement within well-formed sentences from Universal Dependencies (UD) datasets. The core assumption is that these sentences are grammatically correct, and their UPOS and dependency annotations are accurate. The algorithm randomly reorders verb tokens within their respective phrases while preserving noun phrases, ensuring that the resulting sentences are ungrammatical.

## Method
For every sentence containing at least one verb token, the algorithm performs the following steps:

1. **Phrase Extraction**: Identifies and aggregates full verb tokens and their dependent tokens into phrases.
2. **Noun Phrase Identification**: Within these phrases, noun-headed groups of tokens are treated as impermeable units to prevent disruption by verb reordering.
3. **Verb Reordering**: Randomly selects and repositions verb tokens within their respective phrases while keeping the noun phrases intact.
4. **Sentence-Initial Constraints**: If the manipulated phrase is the first in a sentence, verbs are restricted from occupying the initial position to avoid accidentally forming correct polar question syntax in Germanic languages.

## Example
Consider the following input sentence (common punctuation is removed, and tokens are lowercased to eliminate contextual clues beyond word order):

**Original Sentence:**
```
om du inte vill behålla dina filterinställningar kontrollerar du att knappen autofilter inte är markerad innan du börjar markera element som ska filtreras
```

**Step 1: Phrase Extraction**
```
[om du inte vill behålla dina filterinställningar]
[kontrollerar du]
[att knappen autofilter inte är markerad]
[innan du börjar markera element]
[som ska filtreras]
```

**Step 2: Noun Phrase Protection**
```
[om du inte vill behålla (dina filterinställningar)]
[kontrollerar du]
[att knappen autofilter inte är markerad]
[innan du börjar markera element]
[som ska filtreras]
```

**Step 3: Verb Reordering**
```
[om du vill inte behålla (dina filterinställningar)]
[kontrollerar du]
[att markerad knappen är autofilter inte]
[börjar innan du markera element]
[som ska filtreras]
```

**Analysis of Changes:**
1. *1st phrase:* `vill` is misplaced, `behålla` stays in its correct position.
2. *2nd phrase:* `är` and `markerad` are misplaced.
3. *3rd phrase:* `kontrollerar` stays in its correct position.
4. *4th phrase:* `börjar` is misplaced, `markera` stays in its correct position.
5. *5th phrase:* `ska` and `filtreras` stay in their correct positions.

**Output Labeling**

The algorithm labels tokens as follows:
- `O` (Other) - Non-verb tokens
- `C` (Correct) - Correctly placed verb tokens
- `F` (False) - Incorrectly placed verb tokens

```
om du vill  inte  behålla  dina  filterinställningar  kontrollerar   du att   markerad knappen  är autofilter  inte  börjar   innan du markera  element  som   ska   filtreras
O  O  F     O     C        O     O                    C              O  O     F        O        F  O           O     F        O     O  C        O        O     C     C
```

## Label Distribution
The algorithm ensures that the distribution of correctly (`C`) and incorrectly (`F`) placed verbs in each dataset remains approximately equal.
