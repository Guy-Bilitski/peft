from pathlib import Path
from datasets import load_dataset
import evaluate, sacrebleu
from pycocoevalcap.cider.cider import Cider

# ---------- data ----------
ds    = load_dataset("tuetschek/e2e_nlg", split="test")
refs  = [r["human_reference"].strip() for r in ds][:162]
preds = Path("outputs/results/lr_0.0005/system_outputs.txt").read_text().splitlines()
print("preds length: ", len(preds))
print("refs length: ", len(refs))
assert len(preds) == len(refs), "pred/ref length mismatch"

# ---------- BLEU ----------
bleu = sacrebleu.BLEU(tokenize="13a").corpus_score(preds, [refs]).score

# ---------- NIST ----------
nist = evaluate.load("nist_mt").compute(
            predictions=preds,
            references=[[r] for r in refs])["nist_mt"]

# ---------- METEOR & ROUGE-L ----------
meteor = evaluate.load("meteor").compute(
             predictions=preds, references=refs)["meteor"]
rougel = evaluate.load("rouge").compute(
             predictions=preds, references=refs,
             use_stemmer=True)["rougeL"]

# ---------- CIDEr ----------
cider = Cider().compute_score(
            {i:[r] for i, r in enumerate(refs)},
            {i:[h] for i, h in enumerate(preds)})[0]

# ---------- print ----------
print(f"BLEU    : {bleu:6.2f}")
print(f"NIST    : {nist:6.2f}")
print(f"METEOR  : {meteor:6.2f}")
print(f"ROUGE-L : {rougel:6.2f}")
print(f"CIDEr   : {cider:6.2f}")

