import argparse
import os

import pandas as pd
import dataclasses

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=float, default=8.0)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--split", type=str, default="test")

args = parser.parse_args()


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
    df = pd.DataFrame({"src": src, "tgt": tgt, "ec": ec, "ds": ds})
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

    def __repr__(self):
        return f"Non-filter: {self.non_filter:.4f}, Full: {self.full:.4f}, Brenda: {self.brenda:.4f}, Metanetx: {self.metanetx:.4f}, Pathbank: {self.pathbank:.4f}, Rhea: {self.rhea:.4f}, EC1: {self.ec1:.4f}, EC2: {self.ec2:.4f}, EC3: {self.ec3:.4f}, EC4: {self.ec4:.4f}, EC5: {self.ec5:.4f}, EC6: {self.ec6:.4f}, EC7: {self.ec7:.4f}"


train_df = load_df("train")
train_tgt = train_df["tgt"].value_counts()
test_df = load_df(args.split)
test_df["num_train_tgt"] = test_df["tgt"].apply(lambda x: train_tgt.get(x, 0))
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
    all_results_dicts.append(dataclasses.asdict(results))
all_results_df = pd.DataFrame(all_results_dicts)
all_results_df.index = names
print(all_results_df.to_csv("results/all.csv"))
