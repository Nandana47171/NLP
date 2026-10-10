"""NLP-based Abbreviation Expansion System.

Pipeline:  text -> sentence split -> abbreviation detection -> in-document
definition mining -> context-based sense disambiguation (TF-IDF cosine) -> expansion.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent / "data" / "abbreviations.json"
ABBR_RE = re.compile(r"\b[A-Z]{2,6}\b")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
DEF_RE = re.compile(r"((?:[A-Za-z][\w\-]*\s+){1,8}?[A-Za-z][\w\-]*)\s*\(([A-Z]{2,6})\)")


@dataclass
class Expansion:
    abbreviation: str
    expansion: str
    method: str          # dictionary | defined-in-text | context | default-sense
    confidence: float
    start: int
    end: int


def load_dictionary(path=DATA):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mine_definitions(text):
    """Find 'Long Form (ABBR)' patterns whose initials match the abbreviation."""
    found = {}
    for m in DEF_RE.finditer(text):
        words, abbr = m.group(1).split(), m.group(2)
        n = len(abbr)
        cand = words[-n:] if len(words) >= n else []
        initials = "".join(w.replace("-", " ").split()[0][0].upper() for w in cand)
        if cand and initials == abbr:
            found[abbr] = " ".join(cand)
    return found


class AbbreviationExpander:
    def __init__(self, dictionary=None, method="char"):
        self.dict = dictionary or load_dictionary()
        self.method = method  # 'first' | 'word' | 'char'
        self._senses = []
        for ab, senses in self.dict.items():
            for i, s in enumerate(senses):
                self._senses.append((ab, i, f"{s['expansion']} {s['context']}"))
        if method == "word":
            self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        else:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.vec.fit([t for _, _, t in self._senses])
        self.matrix = {}
        for ab, senses in self.dict.items():
            self.matrix[ab] = self.vec.transform([f"{s['expansion']} {s['context']}" for s in senses])

    def _disambiguate(self, abbr, sentence):
        senses = self.dict[abbr]
        if len(senses) == 1:
            return senses[0]["expansion"], "dictionary", 1.0
        if self.method == "first":
            return senses[0]["expansion"], "default-sense", 0.0
        ctx = re.sub(r"\b%s\b" % re.escape(abbr), " ", sentence)
        sims = cosine_similarity(self.vec.transform([ctx.lower()]), self.matrix[abbr])[0]
        if sims.max() <= 0:
            return senses[0]["expansion"], "default-sense", 0.0
        best = int(np.argmax(sims))
        conf = float(sims[best] / (sims.sum() + 1e-9))
        return senses[best]["expansion"], "context", round(conf, 3)

    def expand(self, text):
        local = mine_definitions(text)
        results, offset = [], 0
        for sent in SENT_RE.split(text):
            pos = text.find(sent, offset)
            offset = pos + len(sent)
            for m in ABBR_RE.finditer(sent):
                ab = m.group()
                if ab in local:
                    results.append(Expansion(ab, local[ab], "defined-in-text", 1.0,
                                             pos + m.start(), pos + m.end()))
                elif ab in self.dict:
                    exp, how, conf = self._disambiguate(ab, sent)
                    results.append(Expansion(ab, exp, how, conf, pos + m.start(), pos + m.end()))
        return results

    def expand_text(self, text):
        out, last = [], 0
        defined_spans = [(m.start(2) - 1, m.end(2) + 1) for m in DEF_RE.finditer(text)]
        for r in self.expand(text):
            if any(s <= r.start < e for s, e in defined_spans):
                continue  # keep the author's own definition untouched
            out.append(text[last:r.start])
            out.append(f"{r.expansion} ({r.abbreviation})")
            last = r.end
        out.append(text[last:])
        return "".join(out)
