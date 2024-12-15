class ProteinsManager:
    def __init__(self):
        self.id_to_chunk = {}
        with open("datasets/ecreact/proteins/id_to_chunk.txt", "r") as f:
            for line in f:
                id_, chunk = line.strip().split()
                self.id_to_chunk[id_] = int(chunk)

    def get_chunk(self, id_):
        return self.id_to_chunk[id_]
