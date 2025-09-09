from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz


def normalize_score(score: float) -> float:
    return max(0.0, min(1.0, score))


class HybridRanker:
    def __init__(self, texts: List[str]):
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._texts = texts

    def bm25_scores(self, query: str) -> List[float]:
        return list(self._bm25.get_scores(query.lower().split()))

    def fuzzy_scores(self, query: str) -> List[float]:
        scores: List[float] = []
        for t in self._texts:
            scores.append(fuzz.token_set_ratio(query, t) / 100.0)
        return scores

    def blend(self, dense: List[float], query: str, alpha: float = 0.6) -> List[float]:
        bm = self.bm25_scores(query)
        fuzzy = self.fuzzy_scores(query)
        out: List[float] = []
        for i in range(len(dense)):
            bm_n = normalize_score(bm[i])
            fz_n = normalize_score(fuzzy[i])
            sparse = 0.7 * bm_n + 0.3 * fz_n
            out.append(alpha * dense[i] + (1 - alpha) * sparse)
        return out


