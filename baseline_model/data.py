import random
from collections import Counter
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset
import torch

# ====================================================================
# preprocessing
# ====================================================================
nlp = spacy.load("en_core_web_sm")


def extract_subsentences(text):
    if not text or text.strip() == "":
        return []
    doc = nlp(text.strip())
    findings = []
    for sent in doc.sents:
        s = sent.text.strip()
        s = " ".join(s.split())
        if len(s.split()) >= 3:
            findings.append(s)
    return findings


def deduplicate_sentences(df, max_per_sentence=3, seed=42):
    df = df.copy()
    df['findings'] = df['findings'].fillna('')

    empty_mask = df['findings'] == ''
    df         = df[~empty_mask].reset_index(drop=True)

    all_rows = []
    for img_id, row in tqdm(df.iterrows(), total=len(df),
                             desc='Collecting sentences'):
        text = row['findings'].strip()
        sents = extract_subsentences(text)
        for sent in sents:
            all_rows.append({
                'img_id':     img_id,
                'sentence':   sent,
                'image_path': row['image_path']
            })

    print(f'Before dedup: {len(all_rows)} pairs')

    counts = Counter(r['sentence'] for r in all_rows)
    print('\nTop 10 most common:')
    for sent, cnt in counts.most_common(10):
        print(f'  {cnt:5d}x  {sent[:70]}')

    random.seed(seed)
    random.shuffle(all_rows)

    seen = Counter()
    kept = []
    for row in all_rows:
        sent = row['sentence']
        if seen[sent] < max_per_sentence:
            kept.append(row)
            seen[sent] += 1

    print(f'After  dedup: {len(kept)} pairs')
    print(f'Reduction:    {100*(1-len(kept)/len(all_rows)):.1f}%')

    return kept, df


def build_sentence_dict_from_rows(kept_rows, df, tokenizer, max_length=64):
    all_img_ids                 = df.index.tolist()
    train_img_ids, temp_img_ids = train_test_split(
        all_img_ids, test_size=0.10, random_state=42
    )
    val_img_ids, test_img_ids   = train_test_split(
        temp_img_ids, test_size=0.50, random_state=42
    )

    train_img_set = set(train_img_ids)
    val_img_set   = set(val_img_ids)
    test_img_set  = set(test_img_ids)

    data_dict  = {}
    train_uids = []
    val_uids   = []
    test_uids  = []

    for uid, row in enumerate(tqdm(kept_rows, desc='Building dict')):
        img_id   = row['img_id']
        img_path = row['image_path']

        encoded = tokenizer(
            row['sentence'],
            max_length            = max_length,
            padding               = 'max_length',
            truncation            = True,
            return_attention_mask = True
        )

        data_dict[uid] = {
            'token_ids':      encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'image_path':     img_path,
            'image_id':       img_id,
            'sentence':       row['sentence']
        }

        if img_id in train_img_set:
            train_uids.append(uid)
        elif img_id in val_img_set:
            val_uids.append(uid)
        else:
            test_uids.append(uid)

    print(f'Train: {len(train_uids)} | Val: {len(val_uids)} | Test: {len(test_uids)}')

    return {
        'data_dict':  data_dict,
        'data_split': {
            'train_uids': train_uids,
            'val_uids':   val_uids,
            'test_uids':  test_uids
        },
        'vocab_size': tokenizer.vocab_size
    }

# ====================================================================
# datasets
# ====================================================================
MAX_DIM = 256

train_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1),
    tv.transforms.Resize((MAX_DIM, MAX_DIM)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomRotation(10),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1),
    tv.transforms.Resize((MAX_DIM, MAX_DIM)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5], std=[0.5])
])


class MimicDataset(Dataset):
    def __init__(self, dataset, max_length, transform=train_transform, mode='train'):
        super().__init__()
        self.transform  = transform
        self.mode       = mode
        self.max_length = max_length + 1
        self.datadict   = dataset['data_dict']
        self.vocab_size = dataset['vocab_size']

        if mode == 'train':
            self.keys = dataset['data_split']['train_uids']
        elif mode == 'val':
            self.keys = dataset['data_split']['val_uids']
        elif mode == 'test':
            self.keys = dataset['data_split']['test_uids']
        else:
            raise ValueError(f'mode {mode} not supported')

        self.__sep_id__  = 102
        self.__mask_id__ = 103
        self.__pad_id__  = 0

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        uid   = self.keys[idx]
        entry = self.datadict[uid]

        image = Image.open(entry['image_path']).convert('L')
        if self.transform:
            image = self.transform(image)

        caption = np.array(entry['token_ids'])
        max_len = self.max_length

        max_len_array = np.zeros(max_len, dtype='int')
        cap_mask      = np.zeros(max_len, dtype='int')

        if len(caption) <= max_len:
            cap_mask[:len(caption)]      = 1
            max_len_array[:len(caption)] = caption
        else:
            cap_mask[:]       = 1
            max_len_array     = caption[:max_len]
            max_len_array[-1] = self.__sep_id__

        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)
        image_id = entry['image_id']

        return image, max_len_array, cap_mask, image_id, cap_lens


class MimicDatasetMLM(Dataset):
    def __init__(self, dataset, max_length, transform=train_transform, mode='train'):
        super().__init__()
        self.mode       = mode
        self.max_length = max_length + 1
        self.datadict   = dataset['data_dict']
        self.vocab_size = dataset['vocab_size']

        if mode == 'train':
            self.keys = dataset['data_split']['train_uids']
        elif mode == 'val':
            self.keys = dataset['data_split']['val_uids']
        elif mode == 'test':
            self.keys = dataset['data_split']['test_uids']
        else:
            raise ValueError(f'mode {mode} not supported')

        self.__sep_id__  = 102
        self.__mask_id__ = 103
        self.__pad_id__  = 0

    def __len__(self):
        return len(self.keys)

    def mask_input(self, encoded_texts):
        inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
        inp_mask[encoded_texts <= 3] = False

        labels = -1 * np.ones(encoded_texts.shape, dtype=int)
        labels[inp_mask] = encoded_texts[inp_mask]

        encoded_texts_masked = np.copy(encoded_texts)

        inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
        encoded_texts_masked[inp_mask_2mask] = self.__mask_id__

        inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1/9)
        encoded_texts_masked[inp_mask_2random] = np.random.randint(
            4, self.__mask_id__, inp_mask_2random.sum()
        )

        return encoded_texts_masked, labels

    def __getitem__(self, idx):
        uid   = self.keys[idx]
        entry = self.datadict[uid]

        caption_unmask = np.array(entry['token_ids'])
        caption, label = self.mask_input(caption_unmask)

        max_len       = self.max_length
        max_len_array = np.zeros(max_len, dtype='int')
        cap_mask      = np.zeros(max_len, dtype='int')
        label_full    = -1 * np.ones(max_len, dtype='int')

        if len(caption) <= max_len:
            label_full[:len(label)]      = label
            cap_mask[:len(caption)]      = 1
            max_len_array[:len(caption)] = caption
        else:
            label_full        = label[:max_len]
            label_full[-1]    = -1
            cap_mask[:]       = 1
            max_len_array     = caption[:max_len]
            max_len_array[-1] = self.__sep_id__

        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)
        image_id = entry['image_id']

        return max_len_array, cap_mask, label_full, image_id, cap_lens


def build_dataset_itm(dataset, mode='train', cfg=None, out_dir=None):
    transform = train_transform if mode == 'train' else val_transform
    return MimicDataset(dataset, max_length=cfg.max_length, transform=transform, mode=mode)

def build_dataset_mlm(dataset, mode='train', cfg=None):
    transform = train_transform if mode == 'train' else val_transform
    return MimicDatasetMLM(dataset, max_length=cfg.max_length, transform=transform, mode=mode)


def collate_fn_ignore_none(batch):
    """Drops any None samples (e.g. from a failed image load) before the
    default collate. Used as the DataLoader collate_fn for both ITM and
    MLM loaders."""
    import torch
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# ====================================================================
# samplers
# ====================================================================
class UniqueImageSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_to_uids = {}
        for pos, uid in enumerate(dataset.keys):
            img_id = dataset.datadict[uid]['image_id']
            if img_id not in self.image_to_uids:
                self.image_to_uids[img_id] = []
            self.image_to_uids[img_id].append(pos)  # need position, not uid
        self.image_ids = list(self.image_to_uids.keys())
        print(f'UniqueImageSampler: {len(self.image_ids)} unique images')

    def __iter__(self):
        img_ids = self.image_ids.copy()
        random.shuffle(img_ids)
        selected = []
        for img_id in img_ids:
            uid = random.choice(self.image_to_uids[img_id])
            selected.append(uid)
        return iter(selected)

    def __len__(self):
        return len(self.image_ids)
