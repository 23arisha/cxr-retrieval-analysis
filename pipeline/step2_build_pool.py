import argparse
import json

import config
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from eval_utils.chexbert_utils import load_chexbert, chexbert_label_sentences
from eval_utils.common import CHEXPERT_LABELS
from baseline_model.data import extract_subsentences


def build_candidate_pool(mimic_csv):
    """Extracts every unique findings-section sentence (>=3 words) from
    MIMIC-CXR, with NO deduplication cap (unlike the training-corpus
    preprocessing in src.data.preprocessing.deduplicate_sentences, which
    caps each unique sentence at 3 instances)."""
    mimic = pd.read_csv(mimic_csv)
    mimic["findings"] = mimic["findings"].fillna("")
    mimic = mimic[mimic["findings"] != ""].reset_index(drop=True)

    candidate_pool = set()
    for text in tqdm(mimic["findings"], desc="Extracting sentences"):
        for sent in extract_subsentences(text):
            candidate_pool.add(sent)

    candidate_pool = sorted(candidate_pool)
    print(f"Original candidate pool size: {len(candidate_pool)}")
    return candidate_pool


def print_pool_composition(pool_preds, label_cols, pool_name):
    label_counts = {lbl: 0 for lbl in label_cols}
    for pred in pool_preds:
        for lbl in label_cols:
            if pred[lbl] == 1:
                label_counts[lbl] += 1

    total = len(pool_preds)
    print(f"\n=== {pool_name} pool composition: {total} sentences ===\n")
    print(f"{'Label':<28}{'Count':>8}{'% of pool':>12}")
    for lbl, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total
        print(f"{lbl:<28}{count:>8}{pct:>11.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mimic_csv", default=config.MIMIC_CSV)
    parser.add_argument("--chexbert_repo_dir", default=config.CHEXBERT_REPO,
                         help="Local clone of https://github.com/stanfordmlgroup/CheXbert")
    parser.add_argument("--chexbert_ckpt", default=config.CHEXBERT_CKPT,
                         help="Path to the public chexbert.pth checkpoint")
    parser.add_argument("--output_dir", default=config.ART)
    parser.add_argument("--chexbert_batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_pool = build_candidate_pool(args.mimic_csv)
    with open(os.path.join(args.output_dir, "candidate_pool.json"), "w") as f:
        json.dump(candidate_pool, f)

    cx_mdl, cx_tok = load_chexbert(args.chexbert_repo_dir, args.chexbert_ckpt, device)

    pool_preds = chexbert_label_sentences(
        candidate_pool, cx_mdl, cx_tok, device,
        batch_size=args.chexbert_batch_size, show_progress=True,
    )
    with open(os.path.join(args.output_dir, "pool_preds.json"), "w") as f:
        json.dump(pool_preds, f)
    print_pool_composition(pool_preds, CHEXPERT_LABELS, "Original")


if __name__ == "__main__":
    main()
