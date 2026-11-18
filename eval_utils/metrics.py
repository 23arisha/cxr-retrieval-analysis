"""Ground-truth matrix, micro P/R/F1 and per-finding recall (uncertain = -1 is ignored)."""
import numpy as np
from config import FINDINGS


def build_gt_matrix(eval_df, findings=FINDINGS):
    img_ids_sorted = sorted(row["img_id"] for row in eval_df)
    img_id_to_idx = {img_id: i for i, img_id in enumerate(img_ids_sorted)}
    gt = np.zeros((len(eval_df), len(findings)), dtype=int)
    for row in eval_df:
        idx = img_id_to_idx[row["img_id"]]
        for j, finding in enumerate(findings):
            gt[idx, j] = row[finding]
    return gt, img_id_to_idx


def compute_micro_prf(pred_matrix, gt_matrix):
    valid = gt_matrix != -1
    p, g = pred_matrix[valid], gt_matrix[valid]
    tp = np.sum((p == 1) & (g == 1)); fp = np.sum((p == 1) & (g == 0)); fn = np.sum((p == 0) & (g == 1))
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    return recall, precision, f1


def finding_recall(pred, gt):
    pos = gt == 1                                   # uncertain (-1) labels are not positives
    return (pos & (pred == 1)).sum(0) / np.maximum(pos.sum(0), 1)
