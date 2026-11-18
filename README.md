# Does the Candidate Pool, Not the Model, Explain Missed Findings in CXR Sentence Retrieval?

Code for the preprint *"Comparative Analysis of Clinical Finding Retrieval Difficulty in Chest X-Ray Retrieval Models"*
(Farid & Farooq, Dow University of Health Sciences). <!-- add preprint link -->

**Question.** When a chest X-ray sentence-retrieval system misses a clinically important finding, is the cause the
*model* or the *candidate pool* it searches? We keep three models fixed (our JoImTeRNet-based model, CXR-RePaiR,
CXR-ReDonE) and change only the pool: the original imbalanced pool (129,906 sentences), a finding-balanced pool
(~6.2k sentences), and size-matched random pools as the control.

**Main results** (recall@5, 14 CheXpert findings, 10,000 CheXpert images)
- Balancing raised recall for 13 of 14 findings in all three models; size-matched random pools changed it little and inconsistently.
- Balanced-pool gains exceeded the random-pool range in 36 of 42 finding-model cells.
- Lung Lesion stayed hard under every pool and model, which pool share alone does not explain.
- Balancing trades precision for recall, and "No Finding" recall drops.

## Reproduce

```bash
pip install -r requirements.txt
bash setup_external.sh      # then put chexbert.pth and the CXR-RePaiR checkpoint in external/, and edit config.py
python -m pipeline.step1_build_eval_set    # 10,000 CheXpert images + labels
python -m pipeline.step2_build_pool        # MIMIC-CXR sentences + CheXbert labels (original pool)
python -m pipeline.step3_embed             # embed images and pool once per model
python -m pipeline.step4_run_pools         # original / balanced / random / 30 balanced seeds -> retrieval
python -m pipeline.step5_make_tables       # tables and statistics -> results/
```
A GPU is needed for steps 2-4. Every threshold, seed and pool size lives in `config.py`.

## Layout

```
config.py            paths, findings, seeds, pool sizes (edit the PATHS block)
pipeline/            the experiment, one script per step (1-5)
retrieval_models/    the three models, each exposing encode_images() / encode_texts()
eval_utils/          metrics, CheXbert labeler, CheXpert helpers
baseline_model/      how "our model" was trained (not the paper's contribution)
setup_external.sh    clones CXR-RePaiR, CXR-ReDonE, CheXbert and downloads the ReDonE checkpoint
```

`baseline_model/` is the training code for the JoImTeRNet-based model (ResNet-50 + 3-layer transformer; sentence and
word-region matching, MLM). Run `python -m baseline_model.train` to produce `image_encoder_best.pth` /
`text_encoder_best.pth`, which `retrieval_models/ours.py` loads. Files: `config.py`, `model.py`, `data.py`,
`losses.py`, `trainer.py`, `train.py`.

## Notes
- `step5` prints/saves the seed-42 balanced-pool tables and the 30-replicate summaries (mean +/- SD, % of pools outside
  the random range, hard-finding counts) that the manuscript reports.
- The two baselines are used as released checkpoints with no fine-tuning; models are not meant for architectural comparison.
- Datasets are public (MIMIC-CXR, CheXpert) and not redistributed here.

## Datasets

| Dataset | Role | Source | License |
|---|---|---|---|
| MIMIC-CXR | Training data (30,432 frontal image/report pairs) | https://www.kaggle.com/datasets/zainabhalhoul/mimic-dataset | CC BY-NC 4.0 |
| CheXpert | Evaluation data (10,000-image random sample, seed=42) | https://www.kaggle.com/datasets/ashery/chexpert | CC0: Public Domain |

Both are secondary uses of publicly released, de-identified datasets that
received institutional approval and patient consent waivers at original
release (see manuscript "Ethical standards" section).

## Pretrained checkpoints

| Model | Checkpoint | Source |
|---|---|---|
| CXR-RePaiR | `clip-imp-pretrained_128_6_after_4.pt` | Official CXR-RePaiR repository (Endo et al.) |
| CXR-ReDonE | `checkpoint_59.pth` (trained on MIMIC-PRO, prior-study references removed) | Official CXR-ReDonE repository (Ramesh, Chi, Rajpurkar, 2022) |
| CheXbert | `chexbert.pth` | Official CheXbert repository |
| Our model | `image_encoder_best.pth` / `text_encoder_best.pth`, produced by `baseline_model/train.py` | Trained in this work |

Both CXR-RePaiR and CXR-ReDonE checkpoints were used frozen, without any
additional fine-tuning, in evaluation mode.

## Preprocessing / sentence pools

- Findings-section text was segmented into sentences with spaCy's
  `en_core_web_sm` tokenizer; sentences under 3 words were dropped.
- **Original candidate pool**: 129,906 unique sentences (no dedup cap).
- **Balanced candidate pool**: about 6,2xx unique sentences (6,208 for the seed-42 draw), positive sentences
  per CheXpert finding capped at 450 (seed=42 for subsampling above the
  cap); Pleural Other had the fewest available positives (530) in the full pool.
- **Training-set sentence dedup** (training pipeline only, distinct from the
  retrieval pools above): each unique sentence capped at 3 image-sentence
  instances (seed=42), yielding 148,032 pairs from 221,070 raw pairs.

## Seeds

| Seed | Used for |
|---|---|
| `42` | Training-set dedup, train/val/test split (90/5/5), CheXpert sample draw, representative balanced pool |
| `123` | The 50 size-matched random pools |
| `1000 + i` | The 30 balanced-pool replicates |
| `0` / `7` | Image-level bootstrap (3,000) / finding-level bootstrap (5,000) |

All constants are in `config.py`.

## What each step writes (artifacts/)

| Step | Produces |
|---|---|
| `step1_build_eval_set` | `eval_df.json`, `eval_df_with_paths.json` |
| `step2_build_pool` | `candidate_pool.json`, `pool_preds.json` |
| `step3_embed` | `embeddings.pt` |
| `step4_run_pools` | `pool_runs.pkl` |
| `step5_make_tables` | `results/*.csv` (tables) |
