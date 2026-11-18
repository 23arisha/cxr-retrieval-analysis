import json
import os
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm as tqdm_auto

import config
from config import FINDINGS, MODELS, CAP

OUT = config.ART
candidate_pool = json.load(open(f"{OUT}/candidate_pool.json"))
pool_preds = json.load(open(f"{OUT}/pool_preds.json"))      # CheXbert labels per pool sentence
E = torch.load(f"{OUT}/embeddings.pt", weights_only=False)
ids, img_embs = E["ids"], E["img"]

# ---- balanced pool (representative draw, seed 42) ----
assert len(candidate_pool) == len(pool_preds) == 129906

P = np.array([[p[f] == 1 for f in FINDINGS] for p in pool_preds], dtype=np.uint8)
assert P[:, FINDINGS.index('Pleural Other')].sum() == 530, "pool_preds still has the OLD names"

rng = np.random.default_rng(config.POOL_SEED)
chosen = set()
for j in range(len(FINDINGS)):
    avail = np.flatnonzero(P[:, j])
    pick = avail if len(avail) <= CAP else rng.choice(avail, size=CAP, replace=False)
    chosen.update(int(i) for i in pick)

bal_idx  = np.array(sorted(chosen))
bal_pool = [candidate_pool[i] for i in bal_idx]
bal_idx = np.array(sorted(chosen))
print("Balanced pool size:", len(bal_idx))

# ---- random same-size pools + retrieval in original / balanced / random pools ----
N_POOL = len(candidate_pool)
POOL = E['pool']
for m in MODELS:
    assert POOL[m].shape[0] == N_POOL == 129906, f"{m}: pool embeddings are not the ORIGINAL 129,906-sentence pool"
    assert POOL[m].shape[1] == img_embs[m].shape[1], f"{m}: embedding size mismatch"

# ---- random pools with the SAME size as the balanced pool ----
N_DRAWS = config.N_RANDOM
RANDOM_KIND = 'all'     # 'all' = any pool sentence | 'positives_only' = only sentences with >=1 positive label
population = np.arange(N_POOL) if RANDOM_KIND == 'all' else np.flatnonzero(P.sum(1) > 0)
rng_c = np.random.default_rng(config.RANDOM_SEED)
rand_idx = [np.sort(rng_c.choice(population, size=len(bal_idx), replace=False)) for _ in range(N_DRAWS)]
np.save(f'{OUT}/random_pool_indices_{RANDOM_KIND}.npy', np.array(rand_idx))
print(f"balanced pool: {len(bal_idx)} sentences | {N_DRAWS} random pools ({RANDOM_KIND}) of {len(rand_idx[0])} sentences each")

# ---- GPU top-k search ----
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@torch.no_grad()
def topk_in_pool(img_emb, pool_emb, subset=None, k=10, bs=1000):
    """top-k ORIGINAL pool indices (and scores) for every image, searching only `subset` (None = whole pool)"""
    pe = pool_emb.to(dev)
    if subset is not None:
        pe = pe[torch.as_tensor(np.asarray(subset), device=dev)]
    idx, val = [], []
    for i in range(0, len(img_emb), bs):
        t = (img_emb[i:i+bs].to(dev) @ pe.t()).topk(k, dim=1)
        idx.append(t.indices.cpu()); val.append(t.values.cpu())
    idx, val = torch.cat(idx).numpy(), torch.cat(val).numpy()
    return (idx if subset is None else np.asarray(subset)[idx]), val
# ---- end GPU top-k search ----

def run_pool(subset, keep_scores=False):
    pred, top_, val_ = {}, {}, {}
    for m in MODELS:
        top, val = topk_in_pool(img_embs[m], POOL[m], subset)
        pred[m] = {k: P[top[:, :k]].max(axis=1) for k in (1, 5, 10)}   # labels by lookup in the corrected pool labels
        top_[m] = top.astype(np.int32)
        if keep_scores:
            val_[m] = val
    return pred, top_, val_

PRED, TOP, VAL = {}, {}, {}
PRED['Original'], TOP['Original'], _ = run_pool(None)
PRED['Balanced'], TOP['Balanced'], VAL['Balanced'] = run_pool(bal_idx, keep_scores=True)
RAND_PRED, RAND_TOP = [], []
for s in tqdm_auto(rand_idx, desc="random pools"):
    p_, t_, _ = run_pool(s)
    RAND_PRED.append(p_); RAND_TOP.append(t_)

# ---- 30 balanced-pool replicates (different seeds) ----
N_SEEDS = config.N_BAL_SEEDS
def make_capped(seed, cap=CAP, cols=None):
    rng = np.random.default_rng(1000 + seed)
    ch = set()
    for j in (range(len(FINDINGS)) if cols is None else cols):
        avail = np.flatnonzero(P[:, j])
        pick = avail if len(avail) <= cap else rng.choice(avail, size=cap, replace=False)
        ch.update(int(i) for i in pick)
    return np.array(sorted(ch))

SEED_PRED, SEED_SIZE = [], []
for s in tqdm_auto(range(N_SEEDS), desc="balanced seeds"):
    idx = make_capped(s)
    SEED_SIZE.append(len(idx))
    SEED_PRED.append(run_pool(idx)[0])
print("pool size over seeds:", min(SEED_SIZE), "-", max(SEED_SIZE))

pickle.dump({"PRED": PRED, "RAND_PRED": RAND_PRED, "SEED_PRED": SEED_PRED,
             "bal_idx": bal_idx, "SEED_SIZE": SEED_SIZE}, open(f"{OUT}/pool_runs.pkl", "wb"))
print("saved pool_runs.pkl")
