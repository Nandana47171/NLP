# NLP-based Abbreviation Expansion System

Detects abbreviations in text and expands them to their full forms, using the
surrounding context to choose the right meaning when an abbreviation is ambiguous
(e.g. `PM` = Prime Minister / Post Meridiem / Project Manager).

## How it works
1. **Detection** - regex finds upper-case tokens (2-6 letters) that exist in the dictionary.
2. **In-document definitions** - patterns like `Convolutional Neural Network (CNN)` are mined
   (initials must match) and override the dictionary for that text.
3. **Disambiguation** - the sentence is compared with each candidate sense using TF-IDF
   cosine similarity (character n-grams by default); the best-scoring sense wins.
4. **Fallback** - if there is no overlapping context, the first (most common) sense is used.

## Run
```bash
pip install -r requirements.txt
python main.py --demo
python main.py "The nurse started the IV drip and the HR was 82 bpm."
python main.py --file input.txt --method word      # first | word | char
python evaluate.py                                  # metrics, predictions, chart -> results/
```

## Results (small hand-built dataset, see report)
| Method | Main set (52) | Held-out set (22) |
|---|---|---|
| Baseline (first sense) | 65.4% | 54.5% |
| Word TF-IDF | 98.1% | 63.6% |
| Char n-gram TF-IDF | 100% | 72.7% |

The main set shares vocabulary with the dictionary, so it is optimistic; the held-out set is the fairer estimate.

## Structure
```
abbrev_expander.py   core library
main.py              command-line interface
evaluate.py          evaluation + chart
data/                abbreviation dictionary, test and held-out sentences
results/             metrics.json, predictions CSVs, accuracy chart
```
