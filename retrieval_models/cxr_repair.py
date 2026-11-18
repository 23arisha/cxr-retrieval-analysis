"""CXR-RePaiR (CLIP) - frozen released checkpoint. Needs `git clone` of the official repo (see setup_external.sh)."""
import sys
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import config
sys.path.insert(0, config.REPAIR_REPO)
from clip.model import build_model
from clip import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repair_state_dict = torch.load(config.REPAIR_CKPT, map_location="cpu")
repair_model = build_model(repair_state_dict).float().to(device).eval()

repair_preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    lambda img: img.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def _encode_texts(texts, model, device, batch_size=256):
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc='Encoding candidate pool (RePaiR)'):
            batch = texts[i:i+batch_size]
            toks = clip.tokenize(batch, context_length=model.context_length).to(device)
            emb = model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.cpu())
    return torch.cat(embs, dim=0)

def _encode_images(image_paths, model, device, batch_size=32):
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc='Encoding images (RePaiR)'):
            batch_paths = image_paths[i:i+batch_size]
            imgs = torch.stack([repair_preprocess(Image.open(p).convert('L')) for p in batch_paths]).to(device)
            emb = model.encode_image(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


def encode_texts(texts):
    return _encode_texts(texts, repair_model, device)


def encode_images(paths):
    return _encode_images(paths, repair_model, device, batch_size=64)
