import re

SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(SMILES_TOKENIZER_PATTERN)

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