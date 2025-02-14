import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import random
from tqdm import tqdm
import os
from utils import ProteinsManager, MoleculeManager
from preprocessing.tokenizer_utils import encode_eos_pad
from preprocessing.docking_score import get_reaction_attention_emd

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
                 sample_size=None, shuffle=True, emb_suf=False):
        self.max_length = max_length
        self.emb_suf = emb_suf
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.data = []
        self.samples_ids = []
        self.proteins_manager = ProteinsManager()
        self.molecule_manager = MoleculeManager()
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
        ids_lines = [i for i in range(len(src_lines))]
        if not add_emb:
            emb_lines = [""] * len(src_lines)
            scores_lines = [scores_zero] * len(src_lines)
        else:
            ec_lines = [get_ec_from_src(s) for s in src_lines]
            prot_ids = [self.proteins_manager.get_id(ec) if ec is not None else None for ec in ec_lines]
            emb_lines = [self.proteins_manager.get_emb_file(prot_id, self.emb_suf) if prot_id is not None else None for
                         prot_id in
                         prot_ids]

            if self.emb_suf in ["_re", "_gn1", "_pb1"]:
                scores_lines = [scores_zero] * len(src_lines)
            else:
                scores_lines = [
                    get_reaction_attention_emd(src, self.proteins_manager, self.molecule_manager, tokens=True,
                                               only_src=True)
                    for src in tqdm(src_lines)]
            errors = 0
            if self.emb_suf not in ["_re", "_gn1", "_pb1"]:  # ReactEmed is not sequence - length always 1
                for es_index in range(len(emb_lines)):
                    if emb_lines[es_index] is None or scores_lines[es_index] is None:
                        continue
                    emb = np.load(emb_lines[es_index])[0]
                    if len(emb) != len(scores_lines[es_index]):
                        emb_lines[es_index] = None
                        scores_lines[es_index] = None
                        errors += 1
                print(f"Errors: {errors}")
        assert len(src_lines) == len(tgt_lines) == len(emb_lines) == len(scores_lines)

        if self.sample_size is not None:
            indices = random.sample(range(len(src_lines)), self.sample_size)
            src_lines = [src_lines[i] for i in indices]
            tgt_lines = [tgt_lines[i] for i in indices]
            emb_lines = [emb_lines[i] for i in indices]
            scores_lines = [scores_lines[i] for i in indices]
            ids_lines = [ids_lines[i] for i in indices]

        data = []
        for i in tqdm(range(len(src_lines))):
            if emb_lines[i] is None or scores_lines[i] is None:
                continue
            input_id = encode_eos_pad(self.tokenizer, src_lines[i], self.max_length)
            label = encode_eos_pad(self.tokenizer, tgt_lines[i], self.max_length)
            if input_id is None or label is None:
                continue

            # emb = to_torch_float(emb_lines[i])
            scores = to_torch_float(scores_lines[i])
            data.append(
                {"input_ids": input_id, "labels": label, "emb": np.array([emb_lines[i]]), "docking_scores": scores,
                 "id": torch.tensor([ids_lines[i]])})

        print(f"Dataset {split} loaded, len: {len(data)} / {len(src_lines)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    from preprocessing.tokenizer_utils import TOKENIZER_DIR

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    ds = SeqToSeqDataset(datasets=["ecreact"], split="train", tokenizer=tokenizer, add_emb=[True])
    print(f"EcReAct train dataset len: {len(ds)}")
    uspto = SeqToSeqDataset(datasets=["uspto"], split="train", tokenizer=tokenizer, add_emb=[False])
    print(f"USPTO train dataset len: {len(uspto)}")
