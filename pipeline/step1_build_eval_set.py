import argparse
import config
import os

from eval_utils.common import build_gt_lookups, export_eval_df_json, load_chexpert_eval_set


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chexpert_train_csv", default=config.CHEXPERT_CSV)
    parser.add_argument("--image_root", default=config.CHEXPERT_ROOT,
                         help="Absolute path prefix replacing the CSV's "
                              "'CheXpert-v1.0-small/train/' relative prefix.")
    parser.add_argument("--output_dir", default=config.ART)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_df = load_chexpert_eval_set(
        args.chexpert_train_csv, args.image_root,
        seed=args.seed, n_sample=args.n_sample,
    )

    print(f"Sampled evaluation set size: {len(eval_df)} (seed={args.seed})")
    print("\nPositive cases per finding:")
    from eval_utils.common import CHEXPERT_LABELS
    for lbl in CHEXPERT_LABELS:
        print(f"{lbl:<28} {(eval_df[lbl] == 1).sum():,}")

    # Sanity check: GT lookups build without error and cover every image.
    gt_positive_by_id, gt_usable_by_id = build_gt_lookups(eval_df)
    assert len(gt_positive_by_id) == len(eval_df)
    assert len(gt_usable_by_id) == len(eval_df)

    export_eval_df_json(
        eval_df, os.path.join(args.output_dir, "eval_df.json"), include_path=False)
    export_eval_df_json(
        eval_df, os.path.join(args.output_dir, "eval_df_with_paths.json"), include_path=True)


if __name__ == "__main__":
    main()
