from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math
import sys


class SemanticRanker:
    """
    Rank candidate texts by semantic similarity to a question.

    Backends (auto-detected):
      - 'sbert' (sentence-transformers, if installed)
      - 'tfidf' (scikit-learn, if installed)
      - 'bow' (pure-Python bag-of-words fallback)

    Usage:
      ranker = SemanticRanker(backend="auto")
      results = ranker.rank(question, candidate_texts, top_k=10)

    Each result: { 'index': int, 'text': str, 'score': float }
    """

    def __init__(
        self,
        backend: str = "auto",
        model_name: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.normalize = normalize
        self.backend = self._resolve_backend(backend)
        self.backend_used: str = ""
        self._model = None
        self._vectorizer = None
        self._init_backend(model_name)

    # ---------- Public API ----------
    def rank(
        self,
        question: str,
        candidates: Sequence[str],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Returns candidates sorted by similarity to question (desc).
        """
        clean_cands = [(i, c.strip()) for i, c in enumerate(candidates) if c and c.strip()]
        if not question or not question.strip() or not clean_cands:
            return []

        if self.backend_used == "sbert":
            scores = self._rank_sbert(question, [c for _, c in clean_cands])
        elif self.backend_used == "tfidf":
            scores = self._rank_tfidf(question, [c for _, c in clean_cands])
        else:
            scores = self._rank_bow(question, [c for _, c in clean_cands])

        results = [
            {"index": idx, "text": text, "score": float(score)}
            for (idx, text), score in zip(clean_cands, scores)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            results = results[: max(0, int(top_k))]

        return results

    def prune(
        self,
        question: str,
        candidates: Sequence[str],
        threshold: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split candidates into (kept, dropped) by similarity threshold.
        """
        ranked = self.rank(question, candidates, top_k=None)
        kept = [r for r in ranked if r["score"] >= threshold]
        dropped = [r for r in ranked if r["score"] < threshold]
        return kept, dropped

    # (No triple helpers needed; pass your prepared texts directly.)

    # ---------- Backend selection ----------
    def _resolve_backend(self, backend: str) -> str:
        b = (backend or "auto").lower()
        if b in ("auto", "sbert", "sentence-transformers", "tfidf", "bow"):
            return b
        return "auto"

    def _init_backend(self, model_name: Optional[str]) -> None:
        # Try sbert
        if self.backend in ("auto", "sbert", "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
                self._model = SentenceTransformer(name)
                self.backend_used = "sbert"
                return
            except Exception:
                # Fall through to next backend
                pass

        # Try tfidf
        if self.backend in ("auto", "tfidf"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

                self._vectorizer = TfidfVectorizer()
                self.backend_used = "tfidf"
                return
            except Exception:
                # Fall through to bow
                pass

        # Fallback to pure-Python bag-of-words
        self.backend_used = "bow"

    # ---------- Backends: scoring ----------
    def _rank_sbert(self, question: str, cands: List[str]) -> List[float]:
        assert self._model is not None
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError("NumPy is required for sbert backend.") from e

        q_vec = self._model.encode([question], normalize_embeddings=self.normalize)
        c_vecs = self._model.encode(cands, normalize_embeddings=self.normalize)

        # cosine(q, c) = dot if normalized
        q = q_vec[0]
        scores = (c_vecs @ q).tolist()
        return scores

    def _rank_tfidf(self, question: str, cands: List[str]) -> List[float]:
        assert self._vectorizer is not None
        try:
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        except Exception as e:
            # If sklearn partially missing, fallback to bow
            return self._rank_bow(question, cands)

        corpus = [question] + cands
        X = self._vectorizer.fit_transform(corpus)
        q = X[0]
        C = X[1:]
        sims = cosine_similarity(q, C)[0]
        return sims.tolist()

    def _rank_bow(self, question: str, cands: List[str]) -> List[float]:
        # Simple whitespace tokenization + L2-normalized term count vectors.
        docs = [question] + cands
        tokens_list = [self._tokenize(d) for d in docs]
        vocab = {}
        for tokens in tokens_list:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        def vec(tokens: List[str]) -> List[float]:
            v = [0.0] * len(vocab)
            for t in tokens:
                j = vocab.get(t)
                if j is not None:
                    v[j] += 1.0
            if self.normalize:
                self._l2_normalize_inplace(v)
            return v

        q_vec = vec(tokens_list[0])
        c_vecs = [vec(toks) for toks in tokens_list[1:]]

        scores = [self._cosine(q_vec, c_vec) for c_vec in c_vecs]
        return scores

    # ---------- Utils ----------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Very simple tokenizer: lowercase + split on whitespace
        return text.lower().strip().split()

    @staticmethod
    def _l2_normalize_inplace(v: List[float]) -> None:
        s = math.sqrt(sum(x * x for x in v))
        if s > 0:
            inv = 1.0 / s
            for i in range(len(v)):
                v[i] *= inv

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # If vectors are L2-normalized, this is dot product.
        # But we keep the safe version to work regardless.
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


def example_usage() -> None:
    question = "导致血压升高的主要因素是什么？"
    texts = [
        "高盐饮食导致血压升高",
        "规律运动有助于降低血压",
        "睡眠不足会影响心血管健康",
    ]

    ranker = SemanticRanker(backend="auto")
    ranked = ranker.rank(question, texts)
    print("Backend:", ranker.backend_used)
    for r in ranked:
        print(f"score={r['score']:.3f} | {r['text']}")


if __name__ == "__main__":
    # Quick manual test if executed directly
    example_usage()
