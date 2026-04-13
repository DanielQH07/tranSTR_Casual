import json
from typing import List, Dict, Any


class CausalKnowledgeRetriever:
    def __init__(self, bank_path: str, topk: int = 5):
        self.bank_path = bank_path
        self.topk = topk
        self.entries = self._load_bank(bank_path)

    def _load_bank(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("entries", [])
        return data

    def build_query(self, question: str, candidate: str, video_anchors: List[str], q_family: str) -> str:
        anchors = " ".join(video_anchors or [])
        return f"{question} [CAND] {candidate} [ANCHOR] {anchors} [FAMILY] {q_family}"

    def _keyword_overlap(self, query: str, keywords: List[str]) -> float:
        q_tokens = set(query.lower().split())
        if not keywords:
            return 0.0
        hit = 0
        for kw in keywords:
            if kw.lower() in q_tokens:
                hit += 1
        return hit / max(len(keywords), 1)

    def _relation_bonus(self, relation_type: str, q_family: str) -> float:
        if q_family in {"predictive", "predictive_reason"} and relation_type in {"transition", "intention", "effect"}:
            return 0.15
        if q_family in {"counterfactual", "counterfactual_reason"} and relation_type in {"precondition", "affordance", "dependency"}:
            return 0.15
        return 0.0

    def _family_bias_bonus(self, q_family: str, biases: List[str]) -> float:
        if not biases:
            return 0.0
        return 0.2 if q_family in biases else 0.0

    def score_entry(self, query: str, q_family: str, entry: Dict[str, Any]) -> float:
        overlap = self._keyword_overlap(query, entry.get("keywords", []))
        rel_bonus = self._relation_bonus(entry.get("relation_type", ""), q_family)
        fam_bonus = self._family_bias_bonus(q_family, entry.get("question_family_bias", []))
        return overlap + rel_bonus + fam_bonus

    def retrieve(self, question: str, candidate: str, video_anchors: List[str], q_family: str, topk: int = None):
        k = topk or self.topk
        query = self.build_query(question, candidate, video_anchors, q_family)
        scored = []
        for e in self.entries:
            score = self.score_entry(query, q_family, e)
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, e in scored[:k]:
            results.append(
                {
                    "id": e.get("id"),
                    "text": e.get("text", ""),
                    "relation_type": e.get("relation_type", ""),
                    "score": float(score),
                }
            )
        return results
