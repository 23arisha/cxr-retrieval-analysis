"""CXR-ReDonE (ALBEF) - frozen released checkpoint. Needs the official repo + checkpoint (see setup_external.sh).
The shims below let the 2022 ALBEF code run on current transformers/timm versions."""
import config
from tqdm import tqdm

import torch
import torch.nn as nn
import transformers.modeling_utils as _modeling_utils

# apply_chunking_to_forward (you already added this — kept here for completeness)
if not hasattr(_modeling_utils, "apply_chunking_to_forward"):
    import inspect
    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        tensor_shape = input_tensors[0].shape[chunk_dim]
        num_args = len(inspect.signature(forward_fn).parameters)
        if chunk_size > 0:
            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
            chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
            outputs = tuple(forward_fn(*chunk_set) for chunk_set in zip(*chunks))
            return torch.cat(outputs, dim=chunk_dim)
        return forward_fn(*input_tensors)
    _modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

# find_pruneable_heads_and_indices
if not hasattr(_modeling_utils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index
    _modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

# prune_linear_layer
if not hasattr(_modeling_utils, "prune_linear_layer"):
    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        b = None
        if layer.bias is not None:
            b = (layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach())
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer
    _modeling_utils.prune_linear_layer = prune_linear_layer


import os, sys, types, subprocess
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
import transformers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 1. Clone repo (only need the ALBEF folder) --------------------------
REDONE_DIR = config.REDONE_REPO
ALBEF_DIR = os.path.join(REDONE_DIR, 'ALBEF')


try:
    import transformers.file_utils as _futils
except ModuleNotFoundError:
    _futils = types.ModuleType('transformers.file_utils')
    sys.modules['transformers.file_utils'] = _futils
    transformers.file_utils = _futils

def _noop_decorator_factory(*a, **k):
    def _wrap(fn):
        return fn
    return _wrap

for _name in ['add_code_sample_docstrings', 'add_start_docstrings',
              'add_start_docstrings_to_model_forward', 'replace_return_docstrings']:
    setattr(_futils, _name, _noop_decorator_factory)
if not hasattr(_futils, 'ModelOutput'):
    from transformers.utils import ModelOutput as _ModelOutput
    _futils.ModelOutput = _ModelOutput

# vit.py's register_model decorator
try:
    import timm.models.registry as _tregistry
    if not hasattr(_tregistry, 'register_model'):
        _tregistry.register_model = lambda fn: fn
except ModuleNotFoundError:
    _tregistry = types.ModuleType('timm.models.registry')
    _tregistry.register_model = lambda fn: fn
    sys.modules['timm.models.registry'] = _tregistry

# vit.py's _cfg helper (still present in most timm versions, but just in case)
import timm.models.vision_transformer as _tvit
if not hasattr(_tvit, '_cfg'):
    def _cfg(url='', **kwargs):
        return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), **kwargs}
    _tvit._cfg = _cfg

# ---- 3. Load ALBEF model code (skip the vendored tokenizer entirely) -----
if ALBEF_DIR not in sys.path:
    sys.path.insert(0, ALBEF_DIR)
for m in list(sys.modules):          # drop any stale 'models' module from earlier cells
    if m == 'models' or m.startswith('models.'):
        del sys.modules[m]

from models.model_retrieval import ALBEF as ALBEF_Retrieval
from models.vit import interpolate_pos_embed
from models import xbert as _xbert   # so we can fast-init text_encoder below

_xbert.BertPreTrainedModel.tie_weights = lambda self, *a, **k: None

def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    # newer transformers removed this; xbert never actually uses head_mask here (always None)
    return [None] * num_hidden_layers

_xbert.BertPreTrainedModel.get_head_mask = _get_head_mask

# ---- 5. Build config + model ----------------------------------------------
config = yaml.load(open(os.path.join(ALBEF_DIR, 'configs', 'Retrieval_flickr.yaml')), Loader=yaml.Loader)
config['bert_config'] = os.path.join(ALBEF_DIR, 'configs', 'config_bert.json')
config['image_res'] = 256   # matches the CXR-ReDonE cosine-sim retrieval resolution

redone_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# fast-init: skip downloading pretrained bert-base-uncased weights since the
# checkpoint below overwrites the whole state_dict anyway (same trick as CheXbert)
def _fast_bert_from_pretrained(name, *args, **kwargs):
    cfg = kwargs.get('config')
    add_pooling = kwargs.get('add_pooling_layer', True)
    return _xbert.BertModel(config=cfg, add_pooling_layer=add_pooling)

_orig_xbert_from_pretrained = _xbert.BertModel.from_pretrained
_xbert.BertModel.from_pretrained = _fast_bert_from_pretrained
redone_model = ALBEF_Retrieval(config=config, text_encoder='bert-base-uncased', tokenizer=redone_tokenizer)
_xbert.BertModel.from_pretrained = _orig_xbert_from_pretrained

ckpt = torch.load(config.REDONE_CKPT, map_location='cpu')
state_dict = ckpt['model']
state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], redone_model.visual_encoder)
state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], redone_model.visual_encoder_m)
for key in list(state_dict.keys()):
    if 'bert' in key:
        state_dict[key.replace('bert.', '')] = state_dict.pop(key)

msg = redone_model.load_state_dict(state_dict, strict=False)
redone_model = redone_model.to(device).eval()

# ---- 6. Image preprocessing (matches CXR_ReDonE_module exactly) -----------
# NOTE: raw pixel values 0-255, no ToTensor()/255 scaling — this matches the
# original h5-based pipeline, just adapted to read from a JPG path like your
# own retrieve_topk_sentences does.
redone_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
])

def load_image_for_redone(image_path):
    img = Image.open(image_path).convert('L')
    img = np.array(img, dtype=np.float64)
    img = np.expand_dims(img, axis=0)
    img = np.repeat(img, 3, axis=0)
    img = torch.from_numpy(img).float()
    return redone_transform(img)

@torch.no_grad()
def encode_texts_redone(texts, model, tokenizer, device, batch_size=256, max_length=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding candidate pool (ReDonE)'):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
        text_output = model.text_encoder(enc.input_ids, attention_mask=enc.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state[:, 0, :]
        text_embed = F.normalize(model.text_proj(text_feat), dim=-1)
        embs.append(text_embed.cpu())
    return torch.cat(embs, dim=0)


# official-style padded loader for ReDonE (pad to a 320x320 square, then your redone_transform)
def load_image_for_redone_padded(image_path, desired_size=320):
    img = Image.open(image_path).convert('L')
    ratio = desired_size / max(img.size)
    new_size = tuple(int(x * ratio) for x in img.size)
    img = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new('L', (desired_size, desired_size))
    canvas.paste(img, ((desired_size - new_size[0]) // 2,
                       (desired_size - new_size[1]) // 2))
    arr = np.asarray(canvas, np.float64)
    arr = np.repeat(np.expand_dims(arr, 0), 3, axis=0)
    return redone_transform(torch.from_numpy(arr).float())

@torch.no_grad()
def encode_images_redone_padded(paths, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images (ReDonE, padded)"):
        imgs = torch.stack([load_image_for_redone_padded(p) for p in paths[i:i+batch_size]]).to(device)
        feat = redone_model.visual_encoder(imgs)
        embs.append(F.normalize(redone_model.vision_proj(feat[:, 0, :]), dim=-1).cpu())
    return torch.cat(embs, dim=0)


def encode_texts(texts):
    return encode_texts_redone(texts, redone_model, redone_tokenizer, device).cpu()


def encode_images(paths):
    return encode_images_redone_padded(paths)
