from bioservices import UniProt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
uniprot = UniProt()

def ec_to_id_fasta(ec):
    """
    Fetch the UniProt ID and FASTA sequence for a given EC number.
    """
    ec_parts = ec.split(".")
    if len(ec_parts) < 4 or not all(part.isdigit() for part in ec_parts[:4]):
        print(f"Error: {ec} (not a valid EC number)")
        return "", ""
    try:
        results = uniprot.search(f"ec:{ec}", limit=1, frmt="fasta", size=1)
        results = results.splitlines()
        if len(results) < 2:
            print(f"Error: {ec}")
            return "", ""
        uniprot_id = results[0].split("|")[1]
        fasta = "".join(results[1:])
        print(f"{ec} -> {uniprot_id}")
        return uniprot_id, fasta
    except Exception as e:
        print(f"Error: {ec}, {e}")
        return "", ""

if __name__ == "__main__":
    with open("datasets/ecreact/ec.txt") as f:
        all_ec = f.read().splitlines()

    with Pool(cpu_count()) as pool:
        with tqdm(total=len(all_ec), desc="Processing EC numbers") as pbar:
            results = []
            for result in pool.imap(ec_to_id_fasta, all_ec):
                results.append(result)
                pbar.update()

    all_ids = []
    all_fasta = []
    for uniprot_id, fasta in results:
        all_ids.append(uniprot_id)
        all_fasta.append(fasta)

    # Write the results to files
    with open("datasets/ecreact/ec_fasta.txt", "w") as f:
        for fasta in all_fasta:
            f.write(fasta + "\n")

    with open("datasets/ecreact/ec_ids.txt", "w") as f:
        for uniprot_id in all_ids:
            f.write(uniprot_id + "\n")
