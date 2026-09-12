"""
EmbeddingGenerator — Sentence embeddings + FAISS index builder.

Reads cleaned data from SQLite, encodes with SentenceTransformer, builds FAISS index.
"""

import numpy as np
import sqlite3
import pickle
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Embed cleaned text and build FAISS index for similarity search."""

    def __init__(self, model_name="all-MiniLM-L6-v2", db_path="processed/apple_support.db",
                 index_dir="processed/faiss_index", embed_column="complaint_text_clean", batch_size=256):
        self.model_name = model_name
        self.db_path = Path(db_path)
        self.index_dir = Path(index_dir)
        self.embed_column = embed_column
        self.batch_size = batch_size

        self.model = None
        self.index = None
        self.metadata_df = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        self.metadata_df = pd.read_sql("SELECT * FROM apple_support", conn)
        conn.close()
        return self

    def generate_embeddings(self, texts):
        """Encode texts into L2-normalized embeddings."""
        self._load_model()
        texts = [t if t.strip() else "empty" for t in texts]
        return self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)

    def build_index(self, embeddings):
        """Build FAISS index (IVF for >10K vectors, Flat otherwise)."""
        n_vectors, dim = embeddings.shape

        if n_vectors > 10_000:
            nlist = min(int(n_vectors ** 0.5), 256)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = max(nlist // 5, 1)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        return self

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "apple_support.index"))
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata_df, f)
        return self

    def load(self):
        index_path = self.index_dir / "apple_support.index"
        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}. Run .run() first.")
        self.index = faiss.read_index(str(index_path))
        with open(self.index_dir / "metadata.pkl", "rb") as f:
            self.metadata_df = pickle.load(f)
        return self

    def search(self, query_text, top_k=5):
        """Search index for similar complaints."""
        self._load_model()
        q_emb = self.model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx == -1:
                continue
            row = self.metadata_df.iloc[idx]
            results.append({
                "rank": rank, "score": round(float(score), 4),
                "complaint_text": row["complaint_text"],
                "apple_reply": row["apple_reply"],
                "is_dm": row["is_dm"],
            })
        return results

    def run(self):
        """Full pipeline: load data → embed → build index → save."""
        self.load_data()
        texts = self.metadata_df[self.embed_column].fillna("").tolist()
        self.build_index(self.generate_embeddings(texts))
        self.save()
        return self


if __name__ == "__main__":
    EmbeddingGenerator().run()
