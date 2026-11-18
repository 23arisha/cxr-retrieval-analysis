import json
import os

import torch

import config
from retrieval_models import ours, cxr_repair, cxr_redone


def main():
    pool = json.load(open(os.path.join(config.ART, "candidate_pool.json")))
    recs = json.load(open(os.path.join(config.ART, "eval_df_with_paths.json")))
    path_of = {int(r["img_id"]): r["Path"] for r in recs}
    ids = sorted(path_of)                       # sorted img_id order = row order of the ground-truth matrix
    paths = [path_of[i] for i in ids]
    assert len(ids) == config.N_EVAL_IMAGES

    img, txt = {}, {}
    for name, mod in (("ours", ours), ("repair", cxr_repair), ("redone", cxr_redone)):
        print("==", name)
        img[name] = mod.encode_images(paths)
        txt[name] = mod.encode_texts(pool)
        assert txt[name].shape[0] == len(pool) and txt[name].shape[1] == img[name].shape[1]
    torch.save({"ids": ids, "img": img, "pool": txt}, os.path.join(config.ART, "embeddings.pt"))
    print({m: tuple(e.shape) for m, e in img.items()})


if __name__ == "__main__":
    main()
