"""
IntentClassifier — Cluster-based intent discovery with LLM labeling.

Uses KMeans on FAISS embeddings to discover intent clusters,
then auto-labels each cluster using Groq LLM.
"""

import json
import os
import numpy as np
import faiss
import pickle
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from groq import Groq


class IntentClassifier:
    """Discovers intents from embeddings via clustering + LLM labeling."""

    def __init__(self, index_dir="processed/faiss_index", db_path="processed/apple_support.db",
                 output_dir="processed/intents", n_clusters=12):
        self.index_dir = Path(index_dir)
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.n_clusters = n_clusters

        self.embeddings = None
        self.metadata_df = None
        self.cluster_labels = None
        self.centroids = None
        self.intent_map = {}
        load_dotenv(override=True)

    def load_embeddings(self):
        """Reconstruct all embeddings from the FAISS index."""
        index = faiss.read_index(str(self.index_dir / "apple_support.index"))
        if hasattr(index, "make_direct_map"):
            index.make_direct_map()
        self.embeddings = index.reconstruct_n(0, index.ntotal)

        with open(self.index_dir / "metadata.pkl", "rb") as f:
            self.metadata_df = pickle.load(f)
        return self

    def cluster(self):
        """Run KMeans clustering on embeddings."""
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1024, n_init=10, max_iter=300)
        self.cluster_labels = km.fit_predict(self.embeddings)
        self.centroids = km.cluster_centers_
        return self

    def get_subsamples(self, n_samples=10):
        """Pick N texts closest to each cluster centroid."""
        subsamples = {}
        for cid in range(self.n_clusters):
            indices = np.where(self.cluster_labels == cid)[0]
            sims = self.embeddings[indices] @ self.centroids[cid]
            closest = indices[np.argsort(-sims)[:min(n_samples, len(indices))]]
            subsamples[cid] = self.metadata_df.iloc[closest]["complaint_text"].tolist()
        return subsamples

    def label_with_llm(self, n_samples=10):
        """Auto-label each cluster using Groq LLM."""
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")

        if not api_key:
            self.intent_map = {i: f"cluster_{i}" for i in range(self.n_clusters)}
            return self

        client = Groq(api_key=api_key)
        subsamples = self.get_subsamples(n_samples)

        for cid, texts in subsamples.items():
            numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
            prompt = (
                "You are analyzing customer support messages sent to @AppleSupport on Twitter.\n\n"
                f"Below are {len(texts)} representative messages from the same cluster:\n\n"
                f"{numbered}\n\n"
                "Task: What single customer intent/problem do these messages share?\n\n"
                "Rules:\n"
                "- Reply with ONLY a short intent label (2-4 words, snake_case)\n"
                "- Be specific and concise\n"
                "- Do NOT explain, output only the snake_case label\n\n"
                "Intent label:"
            )
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=250,
                )
                label = (response.choices[0].message.content or "").strip().lower()
                label = label.replace(" ", "_").strip("\"'`.-: \n\r").split("\n")[0].strip()
                self.intent_map[cid] = label or f"cluster_{cid}"
            except Exception:
                self.intent_map[cid] = f"cluster_{cid}"

        return self

    def save(self):
        """Save intent map, centroids, and update SQLite."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "intent_map.json", "w") as f:
            json.dump(self.intent_map, f, indent=2)

        np.save(self.output_dir / "centroids.npy", self.centroids)

        self.metadata_df["cluster_id"] = self.cluster_labels
        self.metadata_df["intent"] = [self.intent_map.get(c, f"cluster_{c}") for c in self.cluster_labels]

        conn = sqlite3.connect(self.db_path)
        self.metadata_df.to_sql("apple_support", conn, if_exists="replace", index=False)
        conn.close()
        return self

    def load(self):
        """Load existing intent map and centroids from disk."""
        with open(self.output_dir / "intent_map.json") as f:
            self.intent_map = {int(k): v for k, v in json.load(f).items()}
        self.centroids = np.load(self.output_dir / "centroids.npy")
        return self

    def classify(self, embedding):
        """Classify a single embedding by nearest centroid. Returns (intent, cluster_id, confidence)."""
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-10)
        norm_centroids = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-10)
        similarities = norm_centroids @ norm_emb
        best = int(np.argmax(similarities))
        return self.intent_map.get(best, f"cluster_{best}"), best, float(similarities[best])

    def plot_clusters(self, save_path=None, sample_size=5000):
        """t-SNE scatter plot of clusters."""
        save_path = save_path or self.output_dir / "cluster_visualization.png"
        n = len(self.embeddings)

        if n > sample_size:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, sample_size, replace=False)
            emb_sample, lbl_sample = self.embeddings[idx], self.cluster_labels[idx]
        else:
            emb_sample, lbl_sample = self.embeddings, self.cluster_labels

        coords = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(emb_sample)
        cmap = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))

        fig, ax = plt.subplots(figsize=(14, 10))
        for cid in range(self.n_clusters):
            mask = lbl_sample == cid
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap[cid]],
                       label=f"{cid}: {self.intent_map.get(cid, f'cluster_{cid}')} ({mask.sum()})", s=8, alpha=0.6)

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=3)
        ax.set_title("Apple Support — Intent Clusters (t-SNE)", fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return self

    def run(self):
        """Full pipeline: load → cluster → label → save → plot."""
        self.load_embeddings()
        self.cluster()
        self.label_with_llm()
        self.save()
        self.plot_clusters()
        return self


if __name__ == "__main__":
    IntentClassifier().run()
