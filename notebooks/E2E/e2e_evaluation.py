from pathlib import Path
from datasets import load_dataset
import evaluate, sacrebleu
from pycocoevalcap.cider.cider import Cider

# ---------- data ----------
ds    = load_dataset("tuetschek/e2e_nlg", split="test")
refs  = [r["human_reference"].strip() for r in ds]
preds = Path("outputs/lora_results/lr_0.0002/system_outputs.txt").read_text().splitlines()
# ds     = load_dataset("tuetschek/e2e_nlg", split="test")
# refs   = [rec["references"] for rec in ds]          # list[list[str]] (len=8 each)
# preds  = Path("outputs/lora_results/system_outputs.txt").read_text().splitlines()


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

# bleu  = sacrebleu.BLEU(tokenize="13a").corpus_score(preds, refs).score
# meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]
# rougel = evaluate.load("rouge").compute(predictions=preds, references=refs, use_stemmer=True)["rougeL"]
# cider  = Cider().compute_score({i:r for i,r in enumerate(refs)},
#                                {i:[p] for i,p in enumerate(preds)})[0]


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


# # E2E-NLG evaluation ──────────────────────────────────────────────────────────
# from pathlib import Path
# from datasets import load_dataset
# import evaluate, sacrebleu
# from pycocoevalcap.cider.cider import Cider        # pip install pycocoevalcap

# # ╭──────────────── DATA ─────────────────╮
# ds     = load_dataset("tuetschek/e2e_nlg", split="test")
# refs = [rec["human_reference"] for rec in ds]
# preds  = Path("outputs/results/lr_0.02/system_outputs.txt").read_text(
#              encoding="utf-8").splitlines()

# print("preds:", len(preds), "refs:", len(refs))
# assert len(preds) == len(refs), "pred/ref length mismatch"

# # ╭──────────────── BLEU (official tokeniser) ─────────────────╮
# #   evaluate has a dataset-specific config that mimics the E2E script
# bleu = evaluate.load("bleu", "e2e_nlg").compute(         # <─ note 2nd arg
#           predictions=preds, references=refs)["bleu"]

# # ╭──────────────── NIST ─────────────────╮
# try:
#     # works if you upgrade to sacrebleu ≥ 3.3
#     from sacrebleu.metrics import NIST
#     nist = NIST().corpus_score(preds, refs).score
# except ImportError:
#     # fallback keeps old sacrebleu; evaluate’s NIST_MT uses the E2E rules too
#     nist = evaluate.load("nist_mt", "e2e_nlg").compute(
#                predictions=preds, references=refs)["nist_mt"]

# # ╭──────────────── METEOR & ROUGE-L (same tokeniser) ─────────╮
# meteor = evaluate.load("meteor", "e2e_nlg").compute(
#              predictions=preds, references=refs)["meteor"]
# rougel = evaluate.load("rouge",  "e2e_nlg").compute(
#              predictions=preds, references=refs)["rougeL"]

# # ╭──────────────── CIDEr ─────────────────╮
# cider = Cider().compute_score(
#             {i: r for i, r in enumerate(refs)},       # gts: list-of-lists
#             {i: [p] for i, p in enumerate(preds)})[0] # res: list-of-1

# # ╭──────────────── PRINT ─────────────────╮
# print(f"BLEU     : {bleu:6.2f}")
# print(f"NIST     : {nist:6.2f}")
# print(f"METEOR   : {meteor:6.2f}")
# print(f"ROUGE-L  : {rougel:6.2f}")
# print(f"CIDEr    : {cider:6.2f}")
