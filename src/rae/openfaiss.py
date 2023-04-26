from typing import Dict, List, Sequence, Tuple

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, d: int):
        self.d = d
        self.db = {}
        self.index = faiss.IndexFlatIP(d)
        self.db["next_id"] = 0

    def add_vectors(self, embeddings: List[Tuple[str, Sequence]], normalize: bool = True):
        next_id = self.db["next_id"]
        keys, vectors = list(zip(*embeddings))

        vectors = np.array(vectors, dtype="float32")
        if normalize:
            faiss.normalize_L2(vectors)

        self.index.add(vectors)

        for key in keys:
            self.db[f"key2faiss_id{key}"] = next_id
            self.db[f"faiss_id2key{next_id}"] = key
            next_id += 1

        self.db["next_id"] = next_id

    def reconstruct(self, key: str):
        return np.array(self.index.reconstruct(self.db[f"key2faiss_id{key}"]))

    def reconstruct_n(self, keys: List[str]):
        return [self.reconstruct(key) for key in keys]

    def search_by_keys(self, query: List[str], k_most_similar) -> Dict[str, Dict[str, float]]:
        query_vectors = np.array(self.reconstruct_n(query))

        retrieved_distances, retrieved_indexes = self.index.search(query_vectors, k_most_similar)

        result = {}
        for query_key, distances, indices in zip(query, retrieved_distances, retrieved_indexes):
            key2distance = {}
            for distance, index in zip(distances, indices):
                if index == -1:
                    continue

                key = self.db[f"faiss_id2key{index}"]
                key2distance[key] = distance

            result[query_key] = key2distance
        return result

    def search_by_vectors(self, query_vectors: np.ndarray, k_most_similar, normalize: bool) -> List[Dict[str, float]]:
        try:
            if normalize:
                faiss.normalize_L2(query_vectors)

            retrieved_distances, retrieved_indexes = self.index.search(query_vectors, k_most_similar)

            result = []
            for distances, indices in zip(retrieved_distances, retrieved_indexes):
                key2distance = {}
                for distance, index in zip(distances, indices):
                    if index == -1:
                        continue

                    key = self.db[f"faiss_id2key{index}"]
                    key2distance[key] = distance

                result.append(key2distance)

            return result
        except Exception as e:
            print(e)
            return None
