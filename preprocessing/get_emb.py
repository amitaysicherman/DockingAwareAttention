import re
import torch
import os
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PortBert:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert").to(device).eval()

    def to_vec(self, seq: str):
        if len(seq) > 1023:
            seq = seq[:1023]
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
        ids = self.tokenizer(seq, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        vec = embedding_repr.last_hidden_state
        return vec.detach().cpu().numpy()


class Esm3MedEmb:
    def __init__(self, size="medium"):
        self.decive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.size = size
        if size == "small":
            self.model = ESMC.from_pretrained("esmc_300m", device=self.decive).eval()
        elif size == "medium":
            self.model = ESMC.from_pretrained("esmc_600m", device=self.decive).eval()
        else:
            raise ValueError(f"Unknown size: {size}")

    def to_vec(self, seq: str):
        if len(seq) > 1023:
            seq = seq[:1023]
        try:
            protein = self.ESMProtein(sequence=seq)
            protein = self.model.encode(protein).to(self.decive)
            conf = self.LogitsConfig(return_embeddings=True, sequence=True)
            vec = self.model.logits(protein, conf).embeddings
            return vec.detach().cpu().numpy()
        except Exception as e:
            print(e)
            return None


class GearNet3Embedder:
    def __init__(self, gearnet_cp_file="preprocessing/mc_gearnet_edge.pth"):

        self.proteins_manager = ProteinsManager()
        self.gearnet_model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=7,
                                            edge_input_dim=59,
                                            num_angle_bin=8,
                                            batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").to(
            device).eval()
        checkpoint = torch.load(gearnet_cp_file)
        self.gearnet_model.load_state_dict(checkpoint)
        self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                                 edge_layers=[
                                                                     geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                                 edge_feature="gearnet")

    def to_vec(self, id: str):
        pdb_file = self.proteins_manager.get_pdb_file(id)
        if pdb_file is None:
            return None
        with open(pdb_file, "r") as f:
            pdb_content = f.read()
        mol = Chem.MolFromPDBBlock(pdb_content, sanitize=False)
        if mol is None:
            return None
        try:
            protein = data.Protein.from_molecule(mol)
        except Exception as e:
            print(e)
            return None
        truncate_transform = transforms.TruncateProtein(max_length=550, random=False)
        protein_view_transform = transforms.ProteinView(view="residue")
        transform = transforms.Compose([truncate_transform, protein_view_transform])
        protein = {"graph": protein}
        protein = transform(protein)
        protein = protein["graph"]
        protein = data.Protein.pack([protein])
        protein = self.graph_construction_model(protein)
        output = self.gearnet_model(protein.to(device), protein.node_feature.float().to(device))
        output = output['node_feature']
        output = output.detach().cpu().numpy()
        # add new dim in axis 0 - (seq_len, 512) -> (1, seq_len, 512)
        output = np.expand_dims(output, axis=0)
        # add zeros in first and last position
        dim = output.shape[-1]
        output = np.concatenate([np.zeros((1, 1, dim)), output, np.zeros((1, 1, dim))], axis=1)
        return output


NAME_TO_EMB_NAME = {
    "esm3": "embeddings_600m.npy",
    "prot_bert": "embeddings_pb.npy",
    "gearnet": "embeddings_gn.npy"
}

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from utils import ProteinsManager

    from utils import ProteinsManager

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="prot_bert", choices=["prot_bert", "esm3", "gearnet"])
    args = parser.parse_args()
    if args.model == "prot_bert":
        from transformers import BertModel, BertTokenizer

        model = PortBert()
    elif args.model == "esm3":
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        model = Esm3MedEmb()
    elif args.model == "gearnet":
        from torchdrug import models, layers, data, transforms
        from torchdrug.layers import geometry
        from rdkit import Chem

        model = GearNet3Embedder()
    else:
        raise ValueError(f"Unknown model: {args.model}")
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
        emb_name = NAME_TO_EMB_NAME[args.model]
        output_emb_file = f"{output_dir}/{emb_name}"
        if os.path.exists(output_emb_file):
            continue
        if args.model == "gearnet":
            embeddings = model.to_vec(id_)
        else:
            embeddings = model.to_vec(sequence)
        if embeddings is None:
            fail_count += 1
            continue
        np.save(output_emb_file, embeddings)
    print(f"Fail count: {fail_count}/{len(ids)}")
