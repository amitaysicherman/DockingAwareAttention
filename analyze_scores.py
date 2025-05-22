import os
from utils import ProteinsManager, MoleculeManager
from preprocessing.tokenizer_utils import tokenize_reaction_smiles
import editdistance
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, ExtraTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import random

from rdkit import Chem
from rdkit.Chem import GraphDescriptors


def get_reaction_docking_confidence(rxn):
    rnx = rxn.replace(" ", "")
    src, ec = rnx.split("|")
    pid = protein_manager.get_id(ec)
    protein_base_dir = protein_manager.get_base_dir(pid)
    if protein_base_dir is None:
        return -100
    mol_ids = [molecule_manager.get_id(mol_smiles) for mol_smiles in src.split(".")]
    scores = []
    for mid in mol_ids:
        if not os.path.exists(f"{protein_base_dir}/{mid}/complex_0/"):
            continue
        mol_file = [x for x in os.listdir(f"{protein_base_dir}/{mid}/complex_0/") if x.startswith("rank1_confidence")]
        if len(mol_file) == 0:
            continue
        mol_file = mol_file[0]
        scores.append(float(mol_file.split("confidence")[-1].replace(".sdf", "")))
    if len(scores) == 0:
        return -100
    return np.max(scores)


def get_reaction_len(tgt):
    return len(tokenize_reaction_smiles(tgt).split(" "))


def get_reaction_src_len(all_src):
    return max([len(tokenize_reaction_smiles(src).split(" ")) for src in all_src.split(".")])


def get_bertz_ct(src: str) -> float:
    v = -1
    for smiles in src.split("."):

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        v = max(v, GraphDescriptors.BertzCT(mol))
    return v


def egt_reaction_edit_distance(src, tgt):
    tgt = tokenize_reaction_smiles(tgt).split(" ")
    return min([editdistance.eval(tokenize_reaction_smiles(src_).split(" "), tgt) for src_ in src.split(".")])


def load_df(split):
    with open(f"datasets/ecreact/src-{split}.txt", "r") as f:
        src_ec = f.read().splitlines()
    src = [x.split("|")[0].replace(" ", "") for x in src_ec]
    ec_full = [x.split("|")[1].strip() for x in src_ec]

    ec = [int(x.split(" ")[0][2:-1]) for x in ec_full]
    with open(f"datasets/ecreact/tgt-{split}.txt", "r") as f:
        tgt = f.read().splitlines()
    tgt = [x.replace(" ", "") for x in tgt]
    with open(f"datasets/ecreact/datasets-{split}.txt", "r") as f:
        ds = f.read().splitlines()
    ds = [x.replace("_reaction_smiles", "") for x in ds]
    assert len(src) == len(tgt) == len(ec) == len(ds), f"{len(src)} {len(tgt)} {len(ec)} {len(ds)}"
    df = pd.DataFrame({"src": src, "tgt": tgt, "ec": ec, "ds": ds, 'rnx': src_ec, 'ec_full': ec_full})
    return df


def analyze_model_performance(df, attr_columns, model1_col_, model2_col_, fit_k=1, predict_k=[1, 3, 5]):
    model1_col = f"{model1_col_}_{fit_k}"
    model2_col = f"{model2_col_}_{fit_k}"
    df = df.dropna(subset=[model1_col, model2_col])
    df['class'] = df.apply(
        lambda row: "DAA" if row[model1_col] == 1 and row[model2_col] == 0 else "PAPER" if row[model1_col] == 0 and row[
            model2_col] == 1 else "SAME", axis=1)
    clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=50, max_features=2)  # "sqrt")
    df_in = df[df['class'] != "SAME"]
    clf.fit(df_in[attr_columns], df_in['class'])
    parts = clf.apply(df[attr_columns])
    df.loc[:, 'part'] = parts

    for k in predict_k:
        model1_col = f"{model1_col_}_{k}"
        model2_col = f"{model2_col_}_{k}"
        print("\nLeaf Statistics:")
        print("Leaf ID | Samples | Model1 Acc | Model2 Acc | Diff | Relative Diff")
        print("-" * 70)

        for part in np.unique(parts):
            df_p = df[df['part'] == part]
            a, b = df_p[model1_col].mean(), df_p[model2_col].mean()

            print(f"Node {part:2d} | {len(df_p):7d} | {a:9.2%} | {b:9.2%} | {(a - b):5.2%} | {(a - b) / b:12.2%} ")

        print("\nClass Labels:", clf.classes_)

        fig, ax = plt.subplots(figsize=(12, 12))
        tree.plot_tree(clf,
                       ax=ax,
                       feature_names=attr_columns,
                       class_names=clf.classes_,
                       filled=True,
                       node_ids=True,  # This adds node numbers to the visualization
                       rounded=True)
        plt.title(f"Decision Tree for k={k}")
        plt.show()


class Args:
    epoch = 8.0
    k = 5
    split = "test"


args = Args()
res_cols = ['ec-ECType.PRETRAINED_daa-4_emb-0.0_ectokens-1',
            'ec-ECType.PAPER_daa-0_emb-0.0_ectokens-0',
            'ec-ECType.NO_EC_daa-0_emb-0.0_ectokens-0',
            "ec-ECType.PRETRAINED_daa-1_emb-0.0_ectokens-1_pb1",
            "ec-ECType.PRETRAINED_daa-1_emb-0.0_ectokens-1_gn1",
            "ec-ECType.PRETRAINED_daa-1_emb-0.0_ectokens-1_re",
            "ec-ECType.PRETRAINED_daa-1_emb-0.0_ectokens-1"]
protein_manager = ProteinsManager()
molecule_manager = MoleculeManager()
train_df = load_df("train")
train_tgt_count = train_df["tgt"].value_counts()
train_src_mols = train_df["src"].apply(lambda x: x.split("."))
train_src_mols_set = set()
for mols in train_src_mols:
    train_src_mols_set.update(mols)

train_src_count = train_df["src"].value_counts()
train_ec_count = train_df["ec_full"].value_counts()

test_df = load_df(args.split)
test_df["src_count"] = test_df["src"].apply(lambda x: train_src_count.get(x, 0))
test_df['mol_in_src'] = test_df['src'].apply(lambda x: all([mol in train_src_mols_set for mol in x.split(".")]))
test_df["tgt_count"] = test_df["tgt"].apply(lambda x: train_tgt_count.get(x, 0))
test_df["good_dock"] = test_df["rnx"].apply(lambda x: get_reaction_docking_confidence(x))
test_df["reaction_len"] = test_df["tgt"].apply(lambda x: get_reaction_len(x))
test_df["reaction_src_len"] = test_df["src"].apply(lambda x: get_reaction_src_len(x))
test_df['berts'] = test_df['src'].apply(get_bertz_ct)
test_df["edit_distance"] = test_df.apply(lambda x: egt_reaction_edit_distance(x["src"], x["tgt"]), axis=1)
test_df["ec_count"] = test_df["ec_full"].apply(lambda x: train_ec_count.get(x, 0))
# convert ec to binary columns
for i in range(1, 7):
    test_df[f"ec_{i}"] = test_df["ec"].apply(lambda x: 1 if x == i else 0)

names = []
all_results_dicts = []
for run_name in res_cols:  # os.listdir("results"):
    for k in [1, 3, 5]:
        file_name = f"results/{run_name}/test_8.0_k{k}.txt"
        if not os.path.exists(file_name):
            continue
        names.append(run_name)
        df = pd.read_csv(file_name, header=None)
        df.set_index(0, inplace=True)
        test_df[f"{run_name}_{k}"] = df[1]

rules = [
    [('reaction_src_len', 100)],
    [('tgt_count', 0)],
    [('src_count', 0)],
    [('edit_distance', 25)],
    [('ec_count', 10)],
    [('ec_count', 5298), ('good_dock', -2.51)],
    [('ec_count', 10), ('good_dock', -2.51)],
    [('edit_distance', 30), ('good_dock', -2.58)],
    [('reaction_src_len', 100), ('good_dock', -2.58)],
    [('tgt_count', 0), ('edit_distance', 25)],
    [('src_count', 0), ('reaction_src_len', 100)],

    [('edit_distance', 25), ('ec_count', 10)],
    [('edit_distance', 25), ('reaction_src_len', 100)],
    [('good_dock', -2.5), ('ec_count', 100), ('reaction_src_len', 100)],

]


def rule_to_str(rule):
    return " & ".join(
        [f"{col} > {val}" if col not in ["tgt_count", "src_count"] else f"{col} <= {val}" for col, val in rule])


for rule in rules:
    print(rule_to_str(rule))
    data = test_df.copy()
    cols_to_use = [f'{c}_{k}' for c in res_cols for k in [1, 3, 5]] + [['tgt_count', 'berts']]
    data = data[[c for c in cols_to_use if c in data.columns]].dropna()
    for col, val in rule:
        if col == "tgt_count" or col == "src_count":
            data = data[data[col] <= val]
        else:
            data = data[data[col] > val]
    all_res = []
    for r_col in res_cols:
        t_res = []
        for k in [1, 3, 5]:
            t_res.append(data[f"{r_col}_{k}"].mean())
        all_res.append(t_res)
    res_df = pd.DataFrame(all_res, columns=[1, 3, 5], index=res_cols)
    print(res_df.to_csv())
ec_cols = [f"ec_{i}" for i in range(1, 7)]
attr_columns = ['src_count', 'tgt_count', 'good_dock', 'reaction_len', 'edit_distance', 'ec_count', "reaction_src_len"]
attr_columns = list(np.random.choice(attr_columns, len(attr_columns), replace=False))
print(attr_columns)

model1_col = 'ec-ECType.PRETRAINED_daa-4_emb-0.0_ectokens-1'
model2_col = 'ec-ECType.PAPER_daa-0_emb-0.0_ectokens-0'
analyze_model_performance(test_df[test_df['src_count'] == 0], attr_columns, model1_col,
                          model2_col, fit_k=3, predict_k=[1, 3, 5])
