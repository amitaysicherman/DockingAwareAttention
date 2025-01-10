import argparse
import os

import pandas as pd
import dataclasses
from utils import ProteinsManager, MoleculeManager

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=float, default=8.0)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--split", type=str, default="test")

args = parser.parse_args()

def get_reaction_docking_confidence(rxn, protein_manager: ProteinsManager, molecule_manager: MoleculeManager):
    rnx = rxn.replace(" ", "")
    src, ec = rnx.split("|")
    pid = protein_manager.get_id(ec)
    protein_base_dir = protein_manager.get_base_dir(pid)
    if protein_base_dir is None:
        return 0, 0, 0

    mol_ids = [molecule_manager.get_id(mol_smiles) for mol_smiles in src.split(".")]
    scores = []
    for mid in mol_ids:
        # mol_file = glob.glob(f"{protein_base_dir}/{mid}/complex_0/rank1_confidence*.sdf")
        if not os.path.exists(f"{protein_base_dir}/{mid}/complex_0/"):
            continue
        mol_file=[x for x in os.listdir(f"{protein_base_dir}/{mid}/complex_0/") if x.startswith("rank1_confidence")]
        if len(mol_file) == 0:
            continue
        mol_file = mol_file[0]
        scores.append(float(mol_file.split("confidence")[-1].replace(".sdf", "")))
    if len(scores) == 0:
        return 0, 0, 0
    return np.mean(scores), min(scores), max(scores)


def load_df(split):
    with open(f"datasets/ecreact/src-{split}.txt", "r") as f:
        src_ec = f.read().splitlines()
    src = [x.split("|")[0].replace(" ", "") for x in src_ec]
    ec = [x.split("|")[1].strip() for x in src_ec]
    ec = [int(x.split(" ")[0][2:-1]) for x in ec]
    with open(f"datasets/ecreact/tgt-{split}.txt", "r") as f:
        tgt = f.read().splitlines()
    tgt = [x.replace(" ", "") for x in tgt]
    with open(f"datasets/ecreact/datasets-{split}.txt", "r") as f:
        ds = f.read().splitlines()
    ds = [x.replace("_reaction_smiles", "") for x in ds]
    assert len(src) == len(tgt) == len(ec) == len(ds), f"{len(src)} {len(tgt)} {len(ec)} {len(ds)}"
    df = pd.DataFrame({"src": src, "tgt": tgt, "ec": ec, "ds": ds, 'rnx': src_ec})
    return df


@dataclasses.dataclass
class Results:
    non_filter: float = 0.0
    full: float = 0.0
    brenda: float = 0.0
    metanetx: float = 0.0
    pathbank: float = 0.0
    rhea: float = 0.0
    ec1: float = 0.0
    ec2: float = 0.0
    ec3: float = 0.0
    ec4: float = 0.0
    ec5: float = 0.0
    ec6: float = 0.0
    ec7: float = 0.0
    good_docking: float = 0.0
    def __repr__(self):
        return f"Non-filter: {self.non_filter:.4f}, Full: {self.full:.4f}, Brenda: {self.brenda:.4f}, Metanetx: {self.metanetx:.4f}, Pathbank: {self.pathbank:.4f}, Rhea: {self.rhea:.4f}, EC1: {self.ec1:.4f}, EC2: {self.ec2:.4f}, EC3: {self.ec3:.4f}, EC4: {self.ec4:.4f}, EC5: {self.ec5:.4f}, EC6: {self.ec6:.4f}, EC7: {self.ec7:.4f}"

protein_manager = ProteinsManager()
molecule_manager = MoleculeManager()
train_df = load_df("train")
train_tgt = train_df["tgt"].value_counts()
test_df = load_df(args.split)
test_df["num_train_tgt"] = test_df["tgt"].apply(lambda x: train_tgt.get(x, 0))
test_df["docking_score_mean"], test_df["docking_score_min"], test_df["docking_score_max"] = zip(
    test_df["rnx"].apply(
        lambda x: get_reaction_docking_confidence(x, protein_manager, molecule_manager)))

names = []
all_results_dicts = []
for run_name in os.listdir("results"):

    file_name = f"results/{run_name}/{args.split}_{args.epoch}_k{args.k}.txt"
    if not os.path.exists(file_name):
        continue
    names.append(run_name)
    df = pd.read_csv(file_name, header=None)
    results = Results()
    results.non_filter = df[1].mean()
    zero_train_tgt = test_df[test_df["num_train_tgt"] == 0].index

    filter_df = df[df[0].isin(zero_train_tgt)]
    results.full = filter_df[1].mean()

    filter_df["ds"] = test_df.loc[filter_df[0], "ds"].values
    results.brenda = filter_df[filter_df["ds"] == "brenda"][1].mean()
    results.metanetx = filter_df[filter_df["ds"] == "metanetx"][1].mean()
    results.pathbank = filter_df[filter_df["ds"] == "pathbank"][1].mean()
    results.rhea = filter_df[filter_df["ds"] == "rhea"][1].mean()

    filter_df["ec"] = test_df.loc[filter_df[0], "ec"].values
    for i in range(1, 8):
        results.__setattr__(f"ec{i}", filter_df[filter_df["ec"] == i][1].mean())

    good_docking = filter_df[filter_df["docking_score_min"] >=-1.5]
    results.good_docking = good_docking[1].mean()
    all_results_dicts.append(dataclasses.asdict(results))
all_results_df = pd.DataFrame(all_results_dicts)
all_results_df.index = names
print(all_results_df.to_csv("results/all.csv"))
