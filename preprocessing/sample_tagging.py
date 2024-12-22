import pandas as pd
from preprocessing.tokenizer_utils import SMILES_REGEX


def load_df(split):
    with open(f"datasets/ecreact/src-{split}.txt", "r") as f:
        src_ec = f.read().splitlines()
    src = [x.split("|")[0].replace(" ", "") for x in src_ec]
    ec = [x.split("|")[1].strip() for x in src_ec]
    with open(f"datasets/ecreact/tgt-{split}.txt", "r") as f:
        tgt = f.read().splitlines()
    tgt = [x.replace(" ", "") for x in tgt]
    with open(f"datasets/ecreact/datasets-{split}.txt", "r") as f:
        ds = f.read().splitlines()
    assert len(src) == len(tgt) == len(ec) == len(ds), f"{len(src)} {len(tgt)} {len(ec)} {len(ds)}"
    df = pd.DataFrame({"src": src, "tgt": tgt, "ec": ec, "ds": ds})
    return df


def ec_to_level(ec, level=1):
    return " ".join(ec.split(" ")[:level])


class SampleTags:
    def __init__(self, split, common_molecules=[], common_ec=[]):
        self.common_molecules = common_molecules
        self.common_ec = common_ec

        self.df = load_df(split)
        self.train_df = load_df("train")
        self.add_all_tag()

    def add_number_of_molecules(self):
        self.df["num_mol"] = self.df["src"].apply(lambda x: len(x.split(".")))

    def add_number_of_large_molecules(self, t=3):
        self.df["num_large_mol"] = self.df["src"].apply(
            lambda x: len([y for y in x.split(".") if len(SMILES_REGEX.findall(y)) >= t]))

    def add_ec_level(self, level):
        self.df[f"ec_l_{level}"] = self.df["ec"].apply(lambda x: ec_to_level(x, level))

    def add_num_train_ec(self, level=1):
        train_ec = self.train_df[f"ec"].apply(lambda x: ec_to_level(x, level)).value_counts()
        curr_ec = self.df["ec"].apply(lambda x: ec_to_level(x, level))
        self.df[f"num_train_ec_{level}"] = curr_ec.apply(lambda x: train_ec.get(x, 0))

    def add_num_train_src(self):
        train_src = self.train_df["src"].value_counts()
        self.df["num_train_src"] = self.df["src"].apply(lambda x: train_src.get(x, 0))

    def add_num_train_tgt(self):
        train_tgt = self.train_df["tgt"].value_counts()
        self.df["num_train_tgt"] = self.df["tgt"].apply(lambda x: train_tgt.get(x, 0))

    def add_most_common_molecules(self, n=10, len_threshold=3):
        if len(self.common_molecules) == 0:

            all_molecules = []
            for x in self.df["src"]:
                all_molecules.extend([y for y in x.split(".") if len(SMILES_REGEX.findall(y)) >= len_threshold])
            all_molecules = pd.Series(all_molecules)
            self.common_molecules = all_molecules.value_counts().head(n).index.tolist()

        for i in range(len(self.common_molecules)):
            print(i, self.common_molecules[i])
            self.df[f"common_mol_{i}"] = self.df["src"].apply(
                lambda x: len([y for y in x.split(".") if y == self.common_molecules[i]]))

    def add_most_common_ec(self, n=10):
        if len(self.common_ec) == 0:
            all_ec = self.df["ec"].copy()
            all_ec = all_ec[all_ec.apply(lambda x: "-" not in x)]
            self.common_ec = all_ec.value_counts().head(n).index.tolist()
        for i in range(len(self.common_ec)):
            print(i, self.common_ec[i])
            self.df[f"common_ec_{i}"] = self.df["ec"].apply(lambda x: x == self.common_ec[i])

    def add_legel_ec(self):
        self.df["legal_ec"] = self.df["ec"].apply(
            lambda x: all([y.replace("[", "").replace("]", "")[1:].isdigit() for y in x.split(" ")]))

    def add_all_tag(self):
        self.add_number_of_molecules()
        self.add_number_of_large_molecules()
        self.add_ec_level(1)
        self.add_ec_level(2)
        self.add_ec_level(3)
        self.add_ec_level(4)

        self.add_num_train_ec(1)
        self.add_num_train_ec(2)
        self.add_num_train_ec(3)
        self.add_num_train_ec(4)
        self.add_num_train_src()
        self.add_num_train_tgt()
        self.add_most_common_molecules(n=50)
        self.add_most_common_ec(n=50)
        self.add_legel_ec()

    def get_query_indexes(self, cols_funcs):
        filtered_df = self.df.copy()
        for col, func in cols_funcs:
            filtered_df = filtered_df[filtered_df[col].apply(func)]
        return filtered_df.index.tolist()


if __name__ == "__main__":
    test = SampleTags("test")
    print(len(test.df))
    print(test.df.head())
    print(test.df.columns)