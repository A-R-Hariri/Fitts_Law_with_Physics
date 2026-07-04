"""
parse_within_results.py
-----------------------
Reads user_sgt/{1..13}/results.txt and extracts acc + bal_acc for the three
within-user models, then prints a summary table.

Models parsed:
  within_mhcnn_raw_base-ft-1
  within_mhcnn_raw_base-ft-5
  within_cnnhcf_raw_base-5

Format of each line in results.txt:
  <model_name>: (tensor(<acc>, ...), <loss>, <bal_acc>, tensor([[...]])
"""

import os, re
import numpy as np
from os.path import join, exists

USER_SGT_ROOT = "user_sgt"
N_USERS       = 13

MODELS = [
    "within_mhcnn_raw_base-ft-1",
    "within_mhcnn_raw_base-ft-5",
    "within_cnnhcf_raw_base-5",
]

# tensor(<value>, ...) or just a plain float
_NUM = r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?"
_TENSOR = rf"tensor\(({_NUM})[^)]*\)"
_FLOAT  = _NUM

# (tensor(<acc>,...), <loss>, <bal_acc>, tensor([[...
LINE_RE = re.compile(
    rf"\((?:{_TENSOR}|({_FLOAT}))\s*,\s*"   # acc — tensor or plain float
    rf"({_FLOAT})\s*,\s*"                    # loss
    rf"(?:{_TENSOR}|({_FLOAT}))\s*,"         # bal_acc — tensor or plain float
)


def parse_results(path):
    """Return dict  model_name -> (acc, bal_acc)  for one results.txt."""
    out = {}
    with open(path) as f:
        text = f.read()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Find which model this line belongs to
        matched_model = None
        for m in MODELS:
            if line.startswith(m + ":"):
                matched_model = m
                break
        if matched_model is None:
            continue

        m = LINE_RE.search(line)
        if m is None:
            print(f"  [WARN] could not parse: {line[:80]}")
            continue

        # Groups: (tensor_acc, plain_acc, loss, tensor_bal, plain_bal)
        acc_t, acc_p, _loss, bal_t, bal_p = m.groups()
        acc     = float(acc_t if acc_t is not None else acc_p)
        bal_acc = float(bal_t if bal_t is not None else bal_p)
        out[matched_model] = (acc, bal_acc)

    return out


def main():
    # Collect per-user results
    user_ids = []
    all_results = {}   # uid -> {model -> (acc, bal_acc)}

    for uid in range(1, N_USERS + 1):
        path = join(USER_SGT_ROOT, str(uid), "results.txt")
        if not exists(path):
            print(f"[SKIP] User {uid}: results.txt not found")
            continue
        parsed = parse_results(path)
        if parsed:
            user_ids.append(uid)
            all_results[uid] = parsed
        else:
            print(f"[WARN] User {uid}: no entries parsed from {path}")

    if not all_results:
        print("No results found. Exiting.")
        return

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Within-user results  (acc / bal_acc, %)   —  {len(user_ids)} users")
    print(sep)

    # Per-model detailed table
    for model in MODELS:
        print(f"\nModel: {model}")
        print(f"  {'User':>6}   {'Acc (%)':>9}   {'BalAcc (%)':>11}")
        print(f"  {'-'*6}   {'-'*9}   {'-'*11}")

        accs, bals = [], []
        for uid in user_ids:
            entry = all_results[uid].get(model)
            if entry is None:
                print(f"  {uid:6d}   {'—':>9}   {'—':>11}")
                continue
            acc, bal = entry
            accs.append(acc * 100)
            bals.append(bal * 100)
            print(f"  {uid:6d}   {acc*100:9.2f}   {bal*100:11.2f}")

        if accs:
            ma, sa = np.mean(accs), np.std(accs, ddof=1)
            mb, sb = np.mean(bals), np.std(bals, ddof=1)
            print(f"  {'MEAN':>6}   {ma:9.2f}   {mb:11.2f}")
            print(f"  {'STD':>6}   {sa:9.2f}   {sb:11.2f}")

    # Summary table across all models
    print(f"\n{sep}")
    print("SUMMARY  (mean ± std across users, %)")
    print(f"{'Model':<35} {'Acc':>14} {'BalAcc':>14}")
    print("-" * 64)
    for model in MODELS:
        accs, bals = [], []
        for uid in user_ids:
            entry = all_results[uid].get(model)
            if entry:
                accs.append(entry[0] * 100)
                bals.append(entry[1] * 100)
        if accs:
            ma, sa = np.mean(accs), np.std(accs, ddof=1)
            mb, sb = np.mean(bals), np.std(bals, ddof=1)
            print(f"{model:<35} {ma:6.2f}±{sa:.2f}  {mb:6.2f}±{sb:.2f}")
        else:
            print(f"{model:<35} {'—':>14} {'—':>14}")
    print(sep)


if __name__ == "__main__":
    main()
