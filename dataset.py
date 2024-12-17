import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from collections import defaultdict
import pandas as pd
import random
from tqdm import tqdm
import os

emb_zero = np.zeros((1, 2560))
scores_zero = np.zeros(1)


def to_torch_float(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    else:
        return torch.tensor(x).float()


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, add_emb, weights=None, max_length=200,
                 sample_size=None, shuffle=True):
        self.max_length = max_length
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.data = []
        self.samples_ids = []

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

        if self.sample_size is not None:
            src_lines, tgt_lines = zip(*random.sample(list(zip(src_lines, tgt_lines)), self.sample_size))

        if add_emb:
            ec_lines = [get_ec_from_seq(text) for text in src_lines]

            uniprot_ids = [self.ec_to_uniprot[ec] if ec in self.ec_to_uniprot else None for ec in ec_lines]
            files_pathed = [f"datasets/docking/{uniprot_id}/protein.npy" for uniprot_id in uniprot_ids]
            emb_lines = [np.load(f) if os.path.exists(f) else None for f in tqdm(files_pathed)]
            scores_lines = [self.doker.dock_src_line(src_lines[i]) if emb_lines is not None else None for i in
                            tqdm(range(len(src_lines)))]

        else:
            emb_lines = [emb_zero] * len(src_lines)
            scores_lines = [scores_zero] * len(src_lines)

        data = []
        for i in tqdm(range(len(src_lines))):
            input_id = encode_eos_pad(self.tokenizer, src_lines[i], self.max_length, no_pad=True, remove_unk=True)
            label = encode_eos_pad(self.tokenizer, tgt_lines[i], self.max_length, no_pad=True, remove_unk=True)
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
    ds = SeqToSeqDataset(datasets=["ecreact/level4"], split="test", tokenizer=t, add_emb=[True])

    print(ds.samples_ids)