import faiss
import json
import numpy as np

class CivicSearcher:
    def __init__(self, index_path, id_list_path, metadata_path):
        # Load the FAISS index
        self.index = faiss.read_index(index_path)

        # Load ID list: position → dataset_id
        with open(id_list_path, "r") as f:
            self.id_list = json.load(f)

        # Load metadata: either a dict {id: {...}} or a list
        with open(metadata_path, "r") as f:
            raw = json.load(f)

        # Normalize metadata into a dict id→info
        if isinstance(raw, dict):
            self.meta = raw
        else:
            # assume list of { "id": ..., "title": ..., ... }
            self.meta = { entry["id"]: entry for entry in raw }

    def search(self, query_vec: np.ndarray, top_k: int = 5, normalize: bool = True):
        """
        query_vec: np.ndarray of shape (1, D), float32
        returns: list of dicts, each dict is metadata + 'score'
        """
        # Optionally normalize (cosine sim)
        if normalize:
            faiss.normalize_L2(query_vec)

        D, I = self.index.search(query_vec, top_k)
        results = []
        for score, pos in zip(D[0], I[0]):
            ds_id = self.id_list[pos]
            info = self.meta.get(ds_id, {}).copy()
            info["score"] = float(score)
            results.append(info)
        return results
