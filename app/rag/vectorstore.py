import os
from chromadb import PersistentClient
from chromadb.utils import embedding_functions


class ChromaVS:
    def __init__(self, persist_dir: str, collection: str = "auto_researcher"):
        self.client = PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
        name=collection,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )


    def add(self, ids, documents, metadatas):
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)


    def query(self, text: str, k: int = 5):
        res = self.collection.query(query_texts=[text], n_results=k)
        hits = []
        for i, _ in enumerate(res["ids"][0]):
            hits.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if "distances" in res else None,
            })
        return hits