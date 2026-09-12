"""
IntentAwareRetriever — Intent-partitioned hybrid retrieval with optional Cohere re-ranking.

Pipeline: Intent Routing → Dense Vector Search → BM25 Lexical → RRF Fusion → Rerank → Context Assembly
"""

import json
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    import cohere
except ImportError:
    cohere = None


class BM25Engine:
    """In-memory BM25 lexical search engine."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.inverted_index = {}   # term -> list of (doc_idx, freq)
        self.idf = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", text) if len(w) > 1]

    def fit(self, texts: List[str]):
        """Build inverted index and compute IDFs."""
        self.corpus_size = len(texts)
        if self.corpus_size == 0:
            return

        total_len = 0
        self.doc_lens = []
        doc_freq = Counter()

        for idx, text in enumerate(texts):
            tokens = self.tokenize(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)

            tf = Counter(tokens)
            for term in tf:
                doc_freq[term] += 1
                self.inverted_index.setdefault(term, []).append((idx, tf[term]))

        self.avg_doc_len = total_len / max(self.corpus_size, 1)

        for term, df in doc_freq.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def score(self, query: str, candidate_indices: Optional[List[int]] = None) -> Dict[int, float]:
        """Compute BM25 scores. Optionally restrict to candidate_indices subset."""
        tokens = self.tokenize(query)
        if not tokens or self.corpus_size == 0:
            return {}

        scores = Counter()
        candidate_set = set(candidate_indices) if candidate_indices is not None else None

        for term in tokens:
            if term not in self.inverted_index:
                continue
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue

            for doc_idx, freq in self.inverted_index[term]:
                if candidate_set is not None and doc_idx not in candidate_set:
                    continue
                doc_len = self.doc_lens[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                scores[doc_idx] += idf * (numerator / (denominator + 1e-9))

        return dict(scores)


class IntentAwareRetriever:
    """
    Retriever for customer support:
      1. Predicts query intent via cluster centroids → narrows search space
      2. Dense vector search within the intent cluster
      3. BM25 keyword search with metadata awareness
      4. Reciprocal Rank Fusion (RRF) for hybrid scoring
      5. Optional Cohere re-ranking for cross-attention refinement
      6. Formats rich context for downstream LLM generation
    """

    def __init__(
        self,
        index_dir: str = "processed/faiss_index",
        db_path: str = "processed/apple_support.db",
        intents_dir: str = "processed/intents",
        model_name: str = "all-MiniLM-L6-v2",
        rerank_model: str = "rerank-v3.5",
    ):
        self.index_dir = Path(index_dir)
        self.db_path = Path(db_path)
        self.intents_dir = Path(intents_dir)
        self.model_name = model_name
        self.rerank_model = rerank_model

        load_dotenv(override=True)
        raw_cohere_key = os.getenv("COHERE_API_KEY", "")
        self.cohere_api_key = raw_cohere_key.strip("\"' \t\r\n")

        # Runtime state (populated by load())
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.centroids: Optional[np.ndarray] = None
        self.intent_map: Dict[int, str] = {}
        self.cluster_to_indices: Dict[int, np.ndarray] = {}
        self.bm25_engine: Optional[BM25Engine] = None
        self.cohere_client = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def load(self):
        """Load FAISS index, metadata, centroids, intent map, and models."""
        if self._initialized:
            return self

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run pipeline first.")

        # 1. SQLite metadata
        conn = sqlite3.connect(self.db_path)
        self.metadata_df = pd.read_sql("SELECT * FROM apple_support", conn)
        conn.close()

        if "is_dm" in self.metadata_df.columns:
            self.metadata_df["is_dm"] = self.metadata_df["is_dm"].astype(str).str.lower().isin(["true", "1", "t"])

        # 2. FAISS embeddings
        index_path = self.index_dir / "apple_support.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}.")

        index = faiss.read_index(str(index_path))
        if hasattr(index, "make_direct_map"):
            index.make_direct_map()
        self.embeddings = index.reconstruct_n(0, index.ntotal)

        # 3. Centroids & intent map
        centroids_path = self.intents_dir / "centroids.npy"
        intent_map_path = self.intents_dir / "intent_map.json"

        if centroids_path.exists() and intent_map_path.exists():
            self.centroids = np.load(centroids_path)
            with open(intent_map_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
                self.intent_map = {int(k): str(v) for k, v in raw_map.items()}
        else:
            self.intent_map = {}
            self.centroids = None

        # 4. Cluster → row indices mapping
        if "cluster_id" in self.metadata_df.columns:
            self.metadata_df["cluster_id"] = pd.to_numeric(
                self.metadata_df["cluster_id"], errors="coerce"
            ).fillna(0).astype(int)
            for cid in self.intent_map:
                mask = self.metadata_df["cluster_id"] == int(cid)
                self.cluster_to_indices[int(cid)] = np.where(mask)[0]

        # 5. BM25 index
        search_texts = []
        for _, row in self.metadata_df.iterrows():
            search_texts.append(
                f"{row.get('complaint_text', '')} {row.get('complaint_text_clean', '')} "
                f"{row.get('apple_reply_clean', '')} {row.get('intent', '')}"
            )
        self.bm25_engine = BM25Engine()
        self.bm25_engine.fit(search_texts)

        # 6. Embedding model
        self.model = SentenceTransformer(self.model_name)

        # 7. Cohere client (optional)
        if self.cohere_api_key and cohere:
            try:
                if hasattr(cohere, "ClientV2"):
                    self.cohere_client = cohere.ClientV2(api_key=self.cohere_api_key)
                else:
                    self.cohere_client = cohere.Client(api_key=self.cohere_api_key)
            except Exception:
                self.cohere_client = None

        self._initialized = True
        return self

    # ------------------------------------------------------------------ #
    #  Intent Prediction
    # ------------------------------------------------------------------ #

    def predict_intent(self, query: str, q_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Classify query intent by cosine similarity to cluster centroids."""
        if not self._initialized:
            self.load()

        if q_emb is None:
            q_emb = self.model.encode(query, normalize_embeddings=True)

        if self.centroids is None or len(self.intent_map) == 0:
            return {"cluster_id": 0, "intent": "unknown", "confidence": 0.0, "top_matches": [], "query_embedding": q_emb}

        similarities = self.centroids @ q_emb
        best_cid = int(np.argmax(similarities))

        sorted_cids = np.argsort(-similarities)[:3]
        top_matches = [
            {"cluster_id": int(cid), "intent": self.intent_map.get(int(cid), f"cluster_{cid}"), "score": float(similarities[cid])}
            for cid in sorted_cids
        ]

        return {
            "cluster_id": best_cid,
            "intent": self.intent_map.get(best_cid, f"cluster_{best_cid}"),
            "confidence": float(similarities[best_cid]),
            "top_matches": top_matches,
            "query_embedding": q_emb,
        }

    # ------------------------------------------------------------------ #
    #  Dense Vector Search (cluster-narrowed)
    # ------------------------------------------------------------------ #

    def vector_search(self, query: str, cluster_id: Optional[int] = None, top_k: int = 25, q_emb: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Cosine similarity search narrowed to a specific intent cluster."""
        if not self._initialized:
            self.load()

        if q_emb is None:
            q_emb = self.model.encode(query, normalize_embeddings=True)

        if cluster_id is None:
            cluster_id = self.predict_intent(query, q_emb=q_emb)["cluster_id"]

        if cluster_id == -1 or cluster_id not in self.cluster_to_indices:
            candidate_indices = np.arange(len(self.embeddings))
            candidate_embs = self.embeddings
        else:
            candidate_indices = self.cluster_to_indices[cluster_id]
            candidate_embs = self.embeddings[candidate_indices]

        if len(candidate_indices) == 0:
            return []

        sims = candidate_embs @ q_emb
        limit = min(top_k, len(candidate_indices))
        best_local = np.argsort(-sims)[:limit]

        return [self._build_doc(int(candidate_indices[li]), rank + 1, float(sims[li]), cluster_id, is_vector=True)
                for rank, li in enumerate(best_local)]

    # ------------------------------------------------------------------ #
    #  BM25 Keyword Search
    # ------------------------------------------------------------------ #

    def keyword_search(self, query: str, cluster_id: Optional[int] = None, top_k: int = 25, prefer_public: bool = False) -> List[Dict[str, Any]]:
        """BM25 lexical search, optionally filtered by cluster and boosted for public solutions."""
        if not self._initialized:
            self.load()

        subset_indices = None
        if cluster_id is not None and cluster_id != -1 and cluster_id in self.cluster_to_indices:
            subset_indices = list(self.cluster_to_indices[cluster_id])

        scores_map = self.bm25_engine.score(query, candidate_indices=subset_indices)
        if not scores_map:
            return []

        sorted_docs = sorted(scores_map.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for rank, (doc_idx, raw_score) in enumerate(sorted_docs, 1):
            row = self.metadata_df.iloc[doc_idx]
            is_dm_val = bool(row.get("is_dm", False))
            adjusted_score = raw_score * 1.25 if (prefer_public and not is_dm_val) else raw_score
            doc = self._row_to_dict(int(doc_idx), row)
            doc.update({"rank_keyword": rank, "keyword_score": float(adjusted_score)})
            results.append(doc)

        return results

    # ------------------------------------------------------------------ #
    #  Hybrid Fusion (RRF)
    # ------------------------------------------------------------------ #

    def hybrid_search(self, query: str, cluster_id: Optional[int] = None, top_k: int = 20, alpha: float = 0.6, rrf_k: int = 60, prefer_public: bool = False) -> List[Dict[str, Any]]:
        """Combine dense vector + BM25 via Reciprocal Rank Fusion."""
        if not self._initialized:
            self.load()

        q_emb = self.model.encode(query, normalize_embeddings=True)
        if cluster_id is None:
            cluster_id = self.predict_intent(query, q_emb=q_emb)["cluster_id"]

        vector_results = self.vector_search(query, cluster_id=cluster_id, top_k=top_k * 2, q_emb=q_emb)
        keyword_results = self.keyword_search(query, cluster_id=cluster_id, top_k=top_k * 2, prefer_public=prefer_public)

        merged: Dict[int, Dict[str, Any]] = {}

        for item in vector_results:
            d_idx = item["doc_idx"]
            merged[d_idx] = item.copy()
            merged[d_idx]["rank_keyword"] = None
            merged[d_idx]["keyword_score"] = 0.0

        for item in keyword_results:
            d_idx = item["doc_idx"]
            if d_idx in merged:
                merged[d_idx]["rank_keyword"] = item["rank_keyword"]
                merged[d_idx]["keyword_score"] = item["keyword_score"]
            else:
                merged[d_idx] = item.copy()
                merged[d_idx]["rank_vector"] = None
                merged[d_idx]["vector_score"] = 0.0

        candidates = []
        for doc in merged.values():
            r_vec, r_kw = doc.get("rank_vector"), doc.get("rank_keyword")
            vec_part = alpha * (1.0 / (rrf_k + r_vec)) if r_vec else 0.0
            kw_part = (1.0 - alpha) * (1.0 / (rrf_k + r_kw)) if r_kw else 0.0
            rrf_score = vec_part + kw_part

            if prefer_public and not doc.get("is_dm", False):
                rrf_score *= 1.15

            doc["rrf_score"] = float(rrf_score)
            candidates.append(doc)

        candidates.sort(key=lambda x: -x["rrf_score"])
        return candidates[:top_k]

    # ------------------------------------------------------------------ #
    #  Cohere Re-Ranking
    # ------------------------------------------------------------------ #

    def rerank_with_cohere(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Re-rank using Cohere cross-encoder. Falls back to RRF ordering."""
        if not candidates:
            return []

        if self.cohere_client:
            doc_texts = []
            for doc in candidates:
                dm_str = "Private DM escalation" if doc.get("is_dm") else "Public troubleshooting solution"
                doc_texts.append(f"Customer Problem: {doc.get('complaint_text', '')}\nApple Resolution [{dm_str}]: {doc.get('apple_reply_clean', '') or doc.get('apple_reply', '')}")

            try:
                response = self.cohere_client.rerank(model=self.rerank_model, query=query, documents=doc_texts, top_n=min(top_k, len(doc_texts)))
                reranked = []
                results = getattr(response, "results", response)
                for new_rank, item in enumerate(results, 1):
                    idx = getattr(item, "index", None) or (item.get("index") if isinstance(item, dict) else None)
                    score = getattr(item, "relevance_score", None) or (item.get("relevance_score", 0.0) if isinstance(item, dict) else 0.0)
                    if idx is not None and 0 <= idx < len(candidates):
                        doc = candidates[idx].copy()
                        doc["rerank_score"] = float(score)
                        doc["rerank_rank"] = new_rank
                        reranked.append(doc)
                if reranked:
                    return reranked
            except Exception:
                pass

        # Fallback: use RRF ordering
        for rank, doc in enumerate(candidates[:top_k], 1):
            doc["rerank_score"] = doc.get("rrf_score", float(1.0 / rank))
            doc["rerank_rank"] = rank
        return candidates[:top_k]

    # ------------------------------------------------------------------ #
    #  Context Assembly for LLM
    # ------------------------------------------------------------------ #

    def build_llm_context(self, query: str, retrieved_docs: List[Dict[str, Any]], intent_info: Optional[Dict[str, Any]] = None) -> str:
        """Format retrieved docs into structured context for the generation model."""
        if not retrieved_docs:
            return "No historical Apple Support resolutions matched this inquiry."

        lines = [
            "### RETRIEVED HIGH-VALUE HISTORICAL RESOLUTIONS (@AppleSupport)",
            f'User Query: "{query}"',
        ]

        if intent_info:
            lines.append(f"Predicted Intent: {intent_info.get('intent', 'general')} (Cluster #{intent_info.get('cluster_id', 0)}) | Confidence: {intent_info.get('confidence', 0.0):.1%}")

        lines.append("")

        for i, doc in enumerate(retrieved_docs, 1):
            is_dm = doc.get("is_dm", False)
            escalation_desc = "PRIVATE DM ESCALATION" if is_dm else "PUBLIC SOLUTION"
            score = doc.get("rerank_score") or doc.get("rrf_score") or 0.0

            lines.append(f"--- [Resolution Example {i}] ---")
            lines.append(f"• Intent: {doc.get('intent', 'customer_support')} (Cluster {doc.get('cluster_id', -1)})")
            lines.append(f"• Type: {escalation_desc} | is_dm: {is_dm}")
            lines.append(f"• Score: {float(score):.4f}")
            lines.append(f'• Customer: "{doc.get("complaint_text", "").strip()}"')
            lines.append(f'• Apple Reply: "{doc.get("apple_reply_clean", "").strip()}"')
            lines.append("")

        lines.append("Guidance: Ground your response in verified solutions above. If DM escalation, indicate private support may be needed.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Full End-to-End Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 3, cluster_id: Optional[int] = None, prefer_public: bool = False, candidate_k: int = 20) -> Dict[str, Any]:
        """Full pipeline: Intent → Hybrid Search → Rerank → Context."""
        if not self._initialized:
            self.load()

        q_emb = self.model.encode(query, normalize_embeddings=True)
        intent_info = self.predict_intent(query, q_emb=q_emb)
        active_cluster = cluster_id if cluster_id is not None else intent_info["cluster_id"]

        candidates = self.hybrid_search(query=query, cluster_id=active_cluster, top_k=candidate_k, prefer_public=prefer_public)
        top_docs = self.rerank_with_cohere(query=query, candidates=candidates, top_k=top_k)
        formatted_context = self.build_llm_context(query=query, retrieved_docs=top_docs, intent_info=intent_info)

        return {
            "query": query,
            "intent_info": intent_info,
            "candidates": candidates,
            "documents": top_docs,
            "formatted_context": formatted_context,
        }

    # ------------------------------------------------------------------ #
    #  Internal Helpers
    # ------------------------------------------------------------------ #

    def _row_to_dict(self, doc_idx: int, row) -> Dict[str, Any]:
        """Convert a metadata DataFrame row to a standard document dict."""
        return {
            "doc_idx": doc_idx,
            "cluster_id": int(row.get("cluster_id", -1)),
            "intent": str(row.get("intent", "")),
            "complaint_text": str(row.get("complaint_text", "")),
            "complaint_text_clean": str(row.get("complaint_text_clean", "")),
            "apple_reply": str(row.get("apple_reply", "")),
            "apple_reply_clean": str(row.get("apple_reply_clean", "")),
            "is_dm": bool(row.get("is_dm", False)),
            "complaint_tweet_id": str(row.get("complaint_tweet_id", "")),
            "reply_tweet_id": str(row.get("reply_tweet_id", "")),
            "created_at_parsed": str(row.get("created_at_parsed", row.get("created_at", ""))),
        }

    def _build_doc(self, global_idx: int, rank: int, score: float, cluster_id: int, is_vector: bool = True) -> Dict[str, Any]:
        """Build a scored document dict from a global row index."""
        row = self.metadata_df.iloc[global_idx]
        doc = self._row_to_dict(global_idx, row)
        if is_vector:
            doc.update({"rank_vector": rank, "vector_score": score})
        else:
            doc.update({"rank_keyword": rank, "keyword_score": score})
        return doc
