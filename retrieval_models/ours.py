"""Our JoImTeRNet-based model. Architecture is imported from baseline_model/ (the code that trained it)."""
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

import config
from baseline_model.model import ImageEncoder, TextEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 512
MAX_DIM = 256

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

val_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1),
    tv.transforms.Resize((MAX_DIM, MAX_DIM)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5], std=[0.5])
])

bert_config = BertConfig(
    vocab_size=tokenizer.vocab_size, hidden_size=HIDDEN_DIM, num_hidden_layers=3,
    num_attention_heads=8, intermediate_size=2048, hidden_act='gelu',
    hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    max_position_embeddings=512, layer_norm_eps=1e-12, initializer_range=0.02,
    type_vocab_size=2, pad_token_id=0
)

image_encoder = ImageEncoder(output_channels=HIDDEN_DIM, pretrained=False).to(DEVICE)
text_encoder = TextEncoder(bert_config, output_channels=HIDDEN_DIM).to(DEVICE)
image_encoder.load_state_dict(torch.load(config.OUR_IMAGE_CKPT, map_location=DEVICE)["model"])
text_encoder.load_state_dict(torch.load(config.OUR_TEXT_CKPT, map_location=DEVICE)["model"])
image_encoder.eval(); text_encoder.eval()


@torch.no_grad()
def encode_texts(texts, batch_size=256):
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding pool (ours)"):
        enc = tokenizer(texts[i:i + batch_size], padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt").to(DEVICE)
        _, sent = text_encoder(enc["input_ids"], enc["attention_mask"], task="itm")
        feats.append(torch.nn.functional.normalize(sent, dim=1).cpu())
    return torch.cat(feats)


@torch.no_grad()
def encode_images(paths, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images (Ours)"):
        imgs = torch.stack([val_transform(Image.open(p).convert('L')) for p in paths[i:i+batch_size]]).to(DEVICE)
        _, global_feat, _, _, _, _ = image_encoder(imgs)
        embs.append(torch.nn.functional.normalize(global_feat, dim=1).cpu())
    return torch.cat(embs, dim=0)
