import re
import torch

SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(SMILES_TOKENIZER_PATTERN)
PAD = "[PAD]"
EOS = "[EOS]"
UNK = "[UNK]"
SPACIAL_TOKENS = {PAD: 0, EOS: 1, UNK: 2}

TOKENIZER_DIR = "datasets/tokenizer"


def tokenize_enzymatic_reaction_smiles(rxn: str) -> str:
    parts = re.split(r">|\|", rxn)
    ec = parts[1].split(".")
    if len(ec) != 4:
        print(f"Error: {rxn} ({len(ec)})")
        return None
    rxn = rxn.replace(f"|{parts[1]}", "")
    tokens = [token for token in SMILES_REGEX.findall(rxn)]
    arrow_index = tokens.index(">>")

    levels = ["v", "u", "t", "q"]

    if ec[0] != "":
        ec_tokens = [f"[{levels[i]}{e}]" for i, e in enumerate(ec)]
        ec_tokens.insert(0, "|")
        tokens[arrow_index:arrow_index] = ec_tokens

    return " ".join(tokens)


def tokenize_reaction_smiles(rxn: str) -> str:
    tokens = [token for token in SMILES_REGEX.findall(rxn)]
    return " ".join(tokens)


def encode_eos_pad(tokenizer, text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if SPACIAL_TOKENS[UNK] in tokens:
        tokens = [x for x in tokens if x != SPACIAL_TOKENS[UNK]]
    tokens = tokens + [tokenizer.eos_token_id]
    if len(tokens) > max_length:
        return None
    return torch.tensor(tokens)


def get_ec_tokens():
    with open(f"{TOKENIZER_DIR}/ec_tokens.txt", "r") as f:
        ec_tokens = f.read().splitlines()
    return ec_tokens
