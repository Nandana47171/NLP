import argparse
from abbrev_expander import AbbreviationExpander

DEMO = ("We train a Convolutional Neural Network (CNN) on drip-chamber images. "
        "The nurse started the IV drip and the monitor showed an HR of 82 bpm. "
        "The PM shared the sprint deadline with the client. "
        "Send the report ASAP, ETA of the delivery is Friday.")

p = argparse.ArgumentParser(description="NLP-based Abbreviation Expansion System")
p.add_argument("text", nargs="?", help="text to expand")
p.add_argument("--file", help="read text from a file")
p.add_argument("--demo", action="store_true")
p.add_argument("--method", choices=["first", "word", "char"], default="char")
a = p.parse_args()

text = DEMO if a.demo else open(a.file, encoding="utf-8").read() if a.file else a.text
if not text:
    p.error("give text, --file, or --demo")
ex = AbbreviationExpander(method=a.method)
print("INPUT :", text, "\n")
print("OUTPUT:", ex.expand_text(text), "\n")
print(f"{'ABBR':6} {'EXPANSION':32} {'METHOD':16} CONF")
for r in ex.expand(text):
    print(f"{r.abbreviation:6} {r.expansion:32} {r.method:16} {r.confidence}")
