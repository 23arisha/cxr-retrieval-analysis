import json
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config
from config import FINDINGS, MODELS, NAME, SHORT
from eval_utils.metrics import build_gt_matrix, compute_micro_prf, finding_recall

OUT = config.RESULTS
ev = json.load(open(os.path.join(config.ART, "eval_df.json")))
gt, idmap = build_gt_matrix(ev)
_R = pickle.load(open(os.path.join(config.ART, "pool_runs.pkl"), "rb"))
PRED, RAND_PRED, SEED_PRED, bal_idx = _R["PRED"], _R["RAND_PRED"], _R["SEED_PRED"], _R["bal_idx"]
N_DRAWS, N_SEEDS = len(RAND_PRED), len(SEED_PRED)


# ======================================================================
# Overall recall / precision / F1 at K = 1, 5, 10 (seed-42 balanced pool)
# ======================================================================
n_img = len(gt)

def prf_cols(pred_by_k):
    out = {}
    for k in (1, 5, 10):
        r, p, f = compute_micro_prf(pred_by_k[k], gt)
        out.update({f'R@{k}': r, f'P@{k}': p, f'F1@{k}': f})
    return out

rows, rows_sd = [], []
for m in MODELS:
    rows.append({'Model': NAME[m], 'Pool': 'Original', **prf_cols(PRED['Original'][m])})
    rows.append({'Model': NAME[m], 'Pool': 'Balanced', **prf_cols(PRED['Balanced'][m])})
    dr = pd.DataFrame([prf_cols(d[m]) for d in RAND_PRED])
    rows.append({'Model': NAME[m], 'Pool': 'Random', **dr.mean().to_dict()})
    rows_sd.append({'Model': NAME[m], 'Pool': 'Random (SD)', **dr.std().to_dict()})
T1 = pd.DataFrame(rows)

print(f"Table 1. Overall retrieval performance (micro-averaged, 14 findings); Random = mean of {N_DRAWS} random pools of the balanced pool's size\n")
print(T1.round(4).to_string(index=False))
print("\nSD across the random pools:")
print(pd.DataFrame(rows_sd).round(4).to_string(index=False))
T1.to_csv(f'{OUT}/table1.csv', index=False)


# ======================================================================
# Per-finding recall@5 (original vs balanced)
# ======================================================================
REC = {pool: {m: finding_recall(PRED[pool][m][5], gt) for m in MODELS} for pool in ('Original', 'Balanced')}
REC_RAND = {m: np.array([finding_recall(d[m][5], gt) for d in RAND_PRED]) for m in MODELS}     # (draws, 14)

rows = []
for j, f in enumerate(FINDINGS):
    row = {'Finding': f}
    for m in MODELS:
        o, b = REC['Original'][m][j], REC['Balanced'][m][j]
        row.update({f'{SHORT[m]}: Orig': o, f'{SHORT[m]}: Bal': b, f'{SHORT[m]}: \u0394': b - o})
    rows.append(row)
T2 = pd.DataFrame(rows)
show = T2.copy()
for c in show.columns:
    if c.endswith('\u0394'):
        show[c] = show[c].map(lambda v: f'{v:+.3f}')
    elif c != 'Finding':
        show[c] = show[c].map(lambda v: f'{v:.3f}')
print("Table 2. Per-finding recall@5 across pools (Orig = original, Bal = balanced, \u0394 = Bal - Orig)\n")
print(show.to_string(index=False))
T2.to_csv(f'{OUT}/table2.csv', index=False)


# ======================================================================
# Recovery: balancing effect vs pool-size effect
# ======================================================================
rows = []
for j, f in enumerate(FINDINGS):
    for m in MODELS:
        d_bal  = REC['Balanced'][m][j] - REC['Original'][m][j]
        d_rand = REC_RAND[m][:, j] - REC['Original'][m][j]               # one value per random pool
        lo, hi = np.percentile(d_rand, [2.5, 97.5])
        rows.append({'Finding': f, 'Model': NAME[m],
                     '\u0394 (Balanced - Original)': f'{d_bal:+.3f}',
                     'Random \u0394 (mean \u00b1 SD)': f'{d_rand.mean():+.3f} \u00b1 {d_rand.std(ddof=1):.3f}',
                     'Random 95% range': f'[{lo:+.3f}, {hi:+.3f}]',
                     'Interpretation': 'Pool-size effect' if lo <= d_bal <= hi else 'Balancing effect'})
T3 = pd.DataFrame(rows)
print("Table 3. Recovery analysis: balancing effect vs pool-size effect (recall@5)")
print("A finding counts as a balancing effect if its \u0394 falls outside the random 95% range.\n")
print(T3.to_string(index=False))
print("\nFindings classified as a balancing effect, per model:")
for m in MODELS:
    sub = T3[(T3['Model'] == NAME[m]) & (T3['Interpretation'] == 'Balancing effect')]
    print(f"  {NAME[m]:<12} {len(sub)}/14  {list(sub['Finding'])}")
T3.to_csv(f'{OUT}/table3.csv', index=False)


# ======================================================================
# Cross-model agreement (Spearman rho)
# ======================================================================
from itertools import combinations
from scipy.stats import rankdata

B = config.N_BOOT_IMAGES                      # bootstrap resamples of the evaluation images
rng_b = np.random.default_rng(0)
Wb  = np.stack([np.bincount(rng_b.integers(0, n_img, n_img), minlength=n_img) for _ in range(B)]).astype(np.float32)
pos = gt == 1
den = np.maximum(Wb @ pos.astype(np.float32), 1)
# SAME resample for both pools; per-finding recall@5 for every resample: (B, 14)
boot = {pool: {m: (Wb @ (pos & (PRED[pool][m][5] == 1)).astype(np.float32)) / den for m in MODELS}
        for pool in ('Original', 'Balanced')}

def spearman_rows(A, Bm):                         # row-wise Spearman rho: (R, F) vs (R, F) -> (R,)
    ra, rb = rankdata(A, axis=1), rankdata(Bm, axis=1)
    ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):          # constant rows give NaN and are ignored below
        return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))

def table4(cols, title):
    rows = []
    for a, b in combinations(MODELS, 2):
        ro = spearman_rows(boot['Original'][a][:, cols], boot['Original'][b][:, cols])
        rb = spearman_rows(boot['Balanced'][a][:, cols], boot['Balanced'][b][:, cols])
        po = spearman_rows(REC['Original'][a][None, cols], REC['Original'][b][None, cols])[0]
        pb = spearman_rows(REC['Balanced'][a][None, cols], REC['Balanced'][b][None, cols])[0]
        rho_rand = spearman_rows(REC_RAND[a][:, cols], REC_RAND[b][:, cols])      # one rho per random pool
        (olo, ohi), (blo, bhi) = np.nanpercentile(ro, [2.5, 97.5]), np.nanpercentile(rb, [2.5, 97.5])
        rows.append({'Model pair': f'{NAME[a]} vs {NAME[b]}',
                     'Original \u03c1 [95% CI]': f'{po:.3f} [{olo:.3f}, {ohi:.3f}]',
                     'Balanced \u03c1 [95% CI]': f'{pb:.3f} [{blo:.3f}, {bhi:.3f}]',
                     'Random \u03c1 (mean \u00b1 SD)': f'{np.nanmean(rho_rand):.3f} \u00b1 {np.nanstd(rho_rand, ddof=1):.3f}'})
    out = pd.DataFrame(rows)
    print(f"\n{title}  (recall@5, {len(cols)} findings; CIs from {B} image-level bootstrap resamples)")
    print(out.to_string(index=False))
    return out

ALL14 = list(range(14))
NO_NF = [j for j, f in enumerate(FINDINGS) if f != 'No Finding']
print("Table 4. Agreement in per-finding difficulty between models (Spearman \u03c1)")
T4_all  = table4(ALL14, "(a) all 14 findings")
T4_noNF = table4(NO_NF, "(b) without 'No Finding' (recommended)")
T4_all.to_csv(f'{OUT}/table4_all14.csv', index=False); T4_noNF.to_csv(f'{OUT}/table4_no_nofinding.csv', index=False)


# ======================================================================
# 30 balanced-pool replicates: mean +/- SD, % outside random range, rho, hard findings
# ======================================================================
rec_seed = {m: np.array([finding_recall(d[m][5], gt) for d in SEED_PRED]) for m in MODELS}  # (seeds,14)

print("Micro metrics, balanced pool, mean ± SD over seeds")
for m in MODELS:
    for k in (1, 5, 10):
        a = np.array([compute_micro_prf(d[m][k], gt) for d in SEED_PRED])
        print(f"{NAME[m]:<11} K={k:<2} R={a[:,0].mean():.4f}±{a[:,0].std(ddof=1):.4f} "
              f"P={a[:,1].mean():.4f}±{a[:,1].std(ddof=1):.4f} F1={a[:,2].mean():.4f}±{a[:,2].std(ddof=1):.4f}")

print("\nShare of seeds where balanced Δ falls outside the random 95% range")
rows = []
for j, f in enumerate(FINDINGS):
    r = {'Finding': f}
    for m in MODELS:
        lo, hi = np.percentile(REC_RAND[m][:, j] - REC['Original'][m][j], [2.5, 97.5])
        ds = rec_seed[m][:, j] - REC['Original'][m][j]
        r[SHORT[m]] = f"{ds.mean():+.3f}±{ds.std(ddof=1):.3f} | {np.mean((ds < lo) | (ds > hi)):.0%}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\nSpearman rho (13 findings) per seed")
for a, b in combinations(MODELS, 2):
    rs = spearman_rows(rec_seed[a][:, NO_NF], rec_seed[b][:, NO_NF])
    print(f"{NAME[a]} vs {NAME[b]}: {np.nanmean(rs):.3f} ± {np.nanstd(rs, ddof=1):.3f} "
          f"[{np.nanpercentile(rs,2.5):.3f}, {np.nanpercentile(rs,97.5):.3f}]")

print("\nHard findings (5 lowest of 13): number of seeds")
rk = {m: rankdata(rec_seed[m][:, NO_NF], axis=1, method='min') for m in MODELS}
hard = {m: rk[m] <= 5 for m in MODELS}
all3 = hard['ours'] & hard['repair'] & hard['redone']
for i, j in enumerate(NO_NF):
    print(f"{FINDINGS[j]:<28} all3: {all3[:,i].sum():>2}/{N_SEEDS} | " +
          " ".join(f"{SHORT[m]}:{hard[m][:,i].sum():>2}" for m in MODELS))


# ======================================================================
# Change in agreement (original - balanced), bootstrap over findings
# ======================================================================
rng_f = np.random.default_rng(7)
cols = np.array(NO_NF)
bal_vec = {m: rec_seed[m].mean(0) for m in MODELS}      # mean over seeds = stable balanced vector
print("Δρ = ρ_original − ρ_balanced, bootstrap over the 13 findings (5000 resamples)")
for a, b in combinations(MODELS, 2):
    d = []
    for _ in range(config.N_BOOT_FINDINGS):
        ix = rng_f.choice(cols, size=len(cols), replace=True)
        ro = spearman_rows(REC['Original'][a][None, ix], REC['Original'][b][None, ix])[0]
        rb = spearman_rows(bal_vec[a][None, ix], bal_vec[b][None, ix])[0]
        d.append(ro - rb)
    d = np.array(d)
    print(f"{NAME[a]} vs {NAME[b]}: Δρ={np.nanmean(d):+.3f} "
          f"[{np.nanpercentile(d,2.5):+.3f}, {np.nanpercentile(d,97.5):+.3f}]")
