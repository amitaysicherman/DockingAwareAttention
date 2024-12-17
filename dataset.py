import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import random
from tqdm import tqdm
import os
from utils import ProteinsManager
from preprocessing.tokenizer_utils import encode_eos_pad
emb_zero = np.zeros((1, 2560))
scores_zero = np.zeros(1)


def to_torch_float(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    else:
        return torch.tensor(x).float()


def get_ec_from_src(text):
    if "|" not in text:
        return None
    return text.split("|")[1]


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, add_emb, weights=None, max_length=200,
                 sample_size=None, shuffle=True):
        self.max_length = max_length
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.data = []
        self.samples_ids = []
        self.proteins_manager = ProteinsManager()
        if weights is None:
            weights = [1] * len(datasets)
        else:
            assert len(weights) == len(datasets)
        assert len(datasets) == len(add_emb)
        for ds, w, ae in zip(datasets, weights, add_emb):
            dataset = self.load_dataset(f"datasets/{ds}", split, ae)
            for _ in range(w):
                self.data.extend(dataset)
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        print(f"Dataset {split} loaded, len: {len(self.data)}")

    def load_dataset(self, input_base, split, add_emb):
        with open(f"{input_base}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()

        with open(f"{input_base}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()
        if not add_emb:
            emb_lines = [emb_zero] * len(src_lines)
            scores_lines = [scores_zero] * len(src_lines)
        else:
            ec_lines = [get_ec_from_src(s) for s in src_lines]
            prot_ids = [self.proteins_manager.get_id(ec) if ec is not None else None for ec in ec_lines]
            emb_files = [self.proteins_manager.get_pdb_file(prot_id) if prot_id is not None else None for prot_id in
                         prot_ids]
            emb_lines = [np.load(f) if os.path.exists(f) else None for f in tqdm(emb_files)]
            scores_file = f"{input_base}/w_{split}.txt"
            if os.path.exists(scores_file):
                with open(scores_file) as f:
                    scores_lines = f.read().splitlines()
                scores_lines = [np.array([float(s) for s in scores.split()]) for scores in scores_lines]
            else:
                scores_lines = [scores_zero] * len(src_lines)
        assert len(src_lines) == len(tgt_lines) == len(emb_lines) == len(scores_lines)

        if self.sample_size is not None:
            indices = random.sample(range(len(src_lines)), self.sample_size)
            src_lines = [src_lines[i] for i in indices]
            tgt_lines = [tgt_lines[i] for i in indices]
            emb_lines = [emb_lines[i] for i in indices]
            scores_lines = [scores_lines[i] for i in indices]

        data = []
        for i in tqdm(range(len(src_lines))):
            input_id = encode_eos_pad(self.tokenizer, src_lines[i], self.max_length)
            label = encode_eos_pad(self.tokenizer, tgt_lines[i], self.max_length)
            emb = to_torch_float(emb_lines[i])
            scores = to_torch_float(scores_lines[i])
            if input_id is None or label is None or emb is None or scores is None:
                continue
            data.append(
                {"input_ids": input_id, "labels": label, "emb": emb, "docking_scores": scores, "id": torch.tensor([i])})

        print(f"Dataset {split} loaded, len: {len(data)} / {len(src_lines)}")
        return data


def __len__(self):
    return len(self.data)


def __getitem__(self, idx):
    return self.data[idx]


if "__main__" == __name__:
    class tok:
        def __init__(self):
            self.eos_token_id = 12
            self.pad_token_id = 0
            self.eos_token_id = 12

        def encode(self, s, **kwargs):
            return [ord(c) for c in s]

        def decode(self, s, **kwargs):
            return "".join([chr(c) for c in s])


    t = tok()
    ds = SeqToSeqDataset(datasets=["ecreact"], split="test", tokenizer=t, add_emb=[True])

    print(ds.samples_ids)
