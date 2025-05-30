from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig
import requests
import numpy as np
from esm.utils.structure.protein_chain import ProteinChain
import os
import torch


def to_pdb(coordinates, sequence, pdb_path):
    return ProteinChain.from_atom37(
        atom37_positions=coordinates,
        id=None,
        sequence=sequence,
        chain_id=None,
        entity_id=None,
        residue_index=None,
        insertion_code=None,
        confidence=None
    ).to_pdb(path=pdb_path)


def fold_protein(sequence, token):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    data = {
        "sequence": sequence,
        "model": "esm3-large-2024-03",
        "potential_sequence_of_concern": False
    }
    api_url = "https://forge.evolutionaryscale.ai/api/v1/fold"
    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        coordinates = response.json()['data']['coordinates']
        coordinates = np.array(coordinates, dtype=np.float32)
        return coordinates
    else:
        print(f"Error: {response.status_code}")
        return None


class ESM3FoldEmbedding:
    def __init__(self, token: str, retry: int = 5, sleep_time: int = 60):
        self.emb_model = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai",
                                                  token=token)
        self.token = token
        self.retry = retry
        self.sleep_time = sleep_time

    def _get_fold_and_embedding(self, protein_seq: str):
        fold = fold_protein(protein_seq, self.token)
        protein = ESMProtein(sequence=protein_seq)

        protein_tensor = self.emb_model.encode(protein)
        logits_output = self.emb_model.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        embeddings = logits_output.embeddings
        return fold, embeddings

    def get_fold_and_embedding(self, protein_seq: str):
        for i in range(self.retry):
            try:
                fold, embeddings = self._get_fold_and_embedding(protein_seq)
                if fold is not None and embeddings is not None:
                    return fold, embeddings
            except Exception as e:
                pass
        return None, None


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from utils import ProteinsManager

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default="")
    args = parser.parse_args()
    assert args.token, "Token is required for ESM3FoldEmbedding"
    model = ESM3FoldEmbedding(token=args.token)

    input_seq_file = "datasets/ecreact/ec_fasta.txt"
    input_ids_file = "datasets/ecreact/ec_ids.txt"
    with open(input_seq_file, "r") as f:
        sequences = f.read().splitlines()
    with open(input_ids_file, "r") as f:
        ids = f.read().splitlines()

    output_base_dir = "datasets/ecreact/proteins/"
    protein_manager = ProteinsManager()
    fail_count = 0
    for id_, sequence in tqdm(zip(ids, sequences), total=len(ids)):

        if len(sequence) == 0:
            continue

        chunk = protein_manager.get_chunk(id_)
        output_dir = f"{output_base_dir}/chunk_{chunk}/{id_}"
        os.makedirs(output_dir, exist_ok=True)
        output_emb_file = f"{output_dir}/embeddings.npy"
        if os.path.exists(output_emb_file):
            continue
        fold, embeddings = model.get_fold_and_embedding(sequence)
        if fold is None or embeddings is None:
            fail_count += 1
            continue
        to_pdb(fold, sequence, f"{output_dir}/fold.pdb")
        np.save(output_emb_file, embeddings)
    print(f"Fail count: {fail_count}/{len(ids)}")
