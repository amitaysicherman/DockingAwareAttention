import os

if __name__ == "__main__":
    import shutil

    n_chunks = 25
    input_seq_file = "datasets/ecreact/ec_fasta.txt"
    input_ids_file = "datasets/ecreact/ec_ids.txt"
    with open(input_seq_file, "r") as f:
        sequences = f.read().splitlines()
    with open(input_ids_file, "r") as f:
        ids = f.read().splitlines()
    output_base_dir = "datasets/ecreact/proteins/"
    ids_to_chunks = [id_ for id_, sequence in zip(ids, sequences) if sequence]
    ids_to_chunks = sorted(ids_to_chunks)
    chunk_size = len(ids_to_chunks) // n_chunks
    id_to_chunk_dict = {}
    for i, id_ in enumerate(ids_to_chunks):
        id_to_chunk_dict[id_] = i // chunk_size
    max_chunk = max(id_to_chunk_dict.values())
    for i in range(max_chunk + 1):
        output_dir = f"{output_base_dir}/chunk_{i}"
        os.makedirs(output_dir, exist_ok=True)

    for id_ in ids:
        path_to_fix = f"{output_base_dir}/{id_}"
        if not os.path.exists(path_to_fix):
            continue
        if id_ not in id_to_chunk_dict:
            continue
        chunk = id_to_chunk_dict[id_]
        new_path = f"{output_base_dir}/chunk_{chunk}/{id_}"
        # move the directory from the old path to the new path
        shutil.move(path_to_fix, new_path)
    # save the mapping
    with open("datasets/ecreact/id_to_chunk.txt", "w") as f:
        for id_, chunk in id_to_chunk_dict.items():
            f.write(f"{id_} {chunk}\n")
