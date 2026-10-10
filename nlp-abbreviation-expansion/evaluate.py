"""Evaluate the abbreviation expander on the labelled test sentences."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abbrev_expander import AbbreviationExpander, load_dictionary

ROOT = Path(__file__).parent
DATASETS = {"main": "test_sentences.json", "heldout": "heldout_sentences.json"}
ALL = {}
dictionary = load_dictionary()
METHODS = {"first": "Baseline (first sense)", "word": "Word TF-IDF", "char": "Char n-gram TF-IDF"}

for ds_name, ds_file in DATASETS.items():
    tests = json.load(open(ROOT / "data" / ds_file, encoding="utf-8"))
    summary, rows = {}, []
    for key, label in METHODS.items():
        ex = AbbreviationExpander(dictionary, method=key)
        tot = cor = amb_tot = amb_cor = det_gold = det_found = 0
        for t in tests:
            pred = {}
            for r in ex.expand(t["text"]):
                pred.setdefault(r.abbreviation, r)
            for ab, gold in t["gold"].items():
                det_gold += 1
                if ab in pred:
                    det_found += 1
                ok = ab in pred and pred[ab].expansion == gold
                tot += 1
                cor += ok
                ambiguous = len(dictionary[ab]) > 1
                amb_tot += ambiguous
                amb_cor += ok and ambiguous
                if key == "char":
                    rows.append([t["text"], ab, gold, pred[ab].expansion if ab in pred else "-",
                                 pred[ab].confidence if ab in pred else 0, "OK" if ok else "WRONG"])
        summary[label] = {
            "mentions": tot, "ambiguous_mentions": amb_tot,
            "overall_accuracy": round(cor / tot, 4),
            "ambiguous_accuracy": round(amb_cor / amb_tot, 4),
            "detection_recall": round(det_found / det_gold, 4),
        }


    ALL[ds_name] = (summary, rows)

(ROOT / "results").mkdir(exist_ok=True)
json.dump({k: v[0] for k, v in ALL.items()}, open(ROOT / "results" / "metrics.json", "w"), indent=2)
for ds_name, (summary, rows) in ALL.items():
    with open(ROOT / "results" / f"predictions_{ds_name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "abbreviation", "gold", "predicted", "confidence", "status"])
        w.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (ds_name, (summary, rows)) in zip(axes, ALL.items()):
    labels = list(summary); x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [summary[l]["overall_accuracy"] * 100 for l in labels], 0.4, label="Overall")
    ax.bar([i + 0.2 for i in x], [summary[l]["ambiguous_accuracy"] * 100 for l in labels], 0.4, label="Ambiguous only")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Main test set" if ds_name == "main" else "Held-out set"); ax.set_ylim(0, 105)
axes[0].set_ylabel("Accuracy (%)"); axes[0].legend()
plt.tight_layout(); plt.savefig(ROOT / "results" / "accuracy_comparison.png", dpi=150)

for ds_name, (summary, rows) in ALL.items():
    print("==", ds_name)
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r[5] == "WRONG":
            print(f"  ERR [{r[1]}] gold={r[2]} | pred={r[3]} | {r[0]}")
