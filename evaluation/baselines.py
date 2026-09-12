"""
Baseline Systems for Apple Support AI Evaluation.

Baseline 1 — Trivial (Random Cluster + Canned Reply):
    Randomly assigns a cluster and returns a fixed generic template.

Baseline 2 — Simple (TF-IDF Nearest Neighbor):
    Uses TF-IDF cosine similarity to find the single most similar historical
    complaint and returns its corresponding apple_reply_clean verbatim.
"""

import random
import re
import sqlite3
from math import log
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


CANNED_REPLY = (
    "Thanks for reaching out! We'd love to look into this for you. "
    "Please send us a DM with your device model and current iOS version "
    "so we can help further: https://support.apple.com"
)


class TrivialBaseline:
    """Random cluster assignment + canned generic reply."""

    def __init__(self, n_clusters: int = 12, seed: int = 42):
        self.n_clusters = n_clusters
        self.rng = random.Random(seed)

    def predict(self, query: str) -> Dict[str, Any]:
        cluster_id = self.rng.randint(0, self.n_clusters - 1)
        return {
            "reply": CANNED_REPLY,
            "predicted_intent": cluster_id,
            "predicted_is_dm": True,  # Always recommends DM
            "confidence": round(self.rng.uniform(0.1, 0.4), 4),
            "top3_intents": [cluster_id,
                             (cluster_id + 1) % self.n_clusters,
                             (cluster_id + 2) % self.n_clusters],
            "guardrail_injection_safe": True,
            "guardrail_valid_query": True,
            "system": "trivial_baseline",
        }


class TFIDFBaseline:
    """
    TF-IDF nearest-neighbor baseline.
    Finds the most similar historical complaint by TF-IDF cosine similarity
    and returns the corresponding Apple reply verbatim.
    """

    def __init__(self, db_path: str = "processed/apple_support.db", max_docs: int = 20000):
        self.db_path = Path(db_path)
        self.max_docs = max_docs
        self.corpus = []       # list of (complaint_text, apple_reply, cluster_id, is_dm)
        self.tfidf_matrix = None
        self.vocab = {}
        self.idf = {}
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9]+\b", text) if len(w) > 1]

    def fit(self):
        """Load corpus from DB and build TF-IDF matrix."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT complaint_text, apple_reply_clean, cluster_id, is_dm FROM apple_support LIMIT ?",
            (self.max_docs,)
        )
        rows = cursor.fetchall()
        conn.close()

        self.corpus = [
            {
                "complaint_text": r[0] or "",
                "apple_reply_clean": r[1] or "",
                "cluster_id": int(r[2]) if r[2] is not None else 0,
                "is_dm": bool(r[3]),
            }
            for r in rows if r[0]
        ]

        # Build vocabulary
        doc_freq = Counter()
        doc_tokens = []
        for doc in self.corpus:
            tokens = self._tokenize(doc["complaint_text"])
            doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freq[t] += 1

        n_docs = len(self.corpus)
        self.vocab = {t: i for i, t in enumerate(doc_freq.keys())}
        self.idf = {t: log((n_docs + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}

        # Build TF-IDF vectors
        n_terms = len(self.vocab)
        self.tfidf_matrix = np.zeros((n_docs, n_terms), dtype=np.float32)

        for doc_idx, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            for term, count in tf.items():
                if term in self.vocab:
                    term_idx = self.vocab[term]
                    self.tfidf_matrix[doc_idx, term_idx] = count * self.idf.get(term, 1.0)

        # Normalize rows
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True) + 1e-10
        self.tfidf_matrix = self.tfidf_matrix / norms
        self._fitted = True

    def _query_vector(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        tf = Counter(tokens)
        for term, count in tf.items():
            if term in self.vocab:
                vec[self.vocab[term]] = count * self.idf.get(term, 1.0)
        norm = np.linalg.norm(vec) + 1e-10
        return vec / norm

    def predict(self, query: str) -> Dict[str, Any]:
        if not self._fitted:
            self.fit()

        q_vec = self._query_vector(query)
        similarities = self.tfidf_matrix @ q_vec
        best_idx = int(np.argmax(similarities))
        best_doc = self.corpus[best_idx]

        return {
            "reply": best_doc["apple_reply_clean"] or CANNED_REPLY,
            "predicted_intent": best_doc["cluster_id"],
            "predicted_is_dm": best_doc["is_dm"],
            "confidence": round(float(similarities[best_idx]), 4),
            "top3_intents": [best_doc["cluster_id"]],
            "guardrail_injection_safe": True,
            "guardrail_valid_query": True,
            "system": "tfidf_baseline",
        }
