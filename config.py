"""Single place for paths, constants and seeds. Edit the PATHS block, nothing else."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")      # intermediate files written by the pipeline
RESULTS = os.path.join(ROOT, "results")    # final tables (csv)
os.makedirs(ART, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

# ------------------------------ PATHS (edit) ------------------------------
def _p(env, default): return os.environ.get(env, default)
CHEXPERT_CSV    = _p("CHEXPERT_CSV",    "data/chexpert/train.csv")
CHEXPERT_ROOT   = _p("CHEXPERT_ROOT",   "data/chexpert/train")
MIMIC_CSV       = _p("MIMIC_CSV",       "data/mimic/mimic-data.csv")
CHEXBERT_REPO   = _p("CHEXBERT_REPO",   "external/CheXbert")
CHEXBERT_CKPT   = _p("CHEXBERT_CKPT",   "external/chexbert.pth")
OUR_IMAGE_CKPT  = _p("OUR_IMAGE_CKPT",  "checkpoints/image_encoder_best.pth")
OUR_TEXT_CKPT   = _p("OUR_TEXT_CKPT",   "checkpoints/text_encoder_best.pth")
REPAIR_REPO     = _p("REPAIR_REPO",     "external/CXR-RePaiR")
REPAIR_CKPT     = _p("REPAIR_CKPT",     "external/clip-imp-pretrained_128_6_after_4.pt")
REDONE_REPO     = _p("REDONE_REPO",     "external/CXR-ReDonE")
REDONE_CKPT     = _p("REDONE_CKPT",     "external/checkpoint_59.pth")

# ------------------------------ CONSTANTS ------------------------------
FINDINGS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
MODELS = ["ours", "repair", "redone"]
NAME   = {"ours": "Our model", "repair": "CXR-RePaiR", "redone": "CXR-ReDonE"}
SHORT  = {"ours": "Our", "repair": "RePaiR", "redone": "ReDonE"}

N_EVAL_IMAGES = 10000
CAP = 450               # max positive sentences per finding in a balanced pool
POOL_SEED = 42          # the "representative" balanced pool (Figure 1)
N_BAL_SEEDS = 30        # balanced-pool replicates
N_RANDOM = 50           # size-matched random pools
RANDOM_SEED = 123
N_BOOT_IMAGES = 3000    # image-level bootstrap (Table 5, original pool)
N_BOOT_FINDINGS = 5000  # finding-level bootstrap (delta rho)
