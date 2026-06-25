from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence

REJECT_TITLES = {
    "civil engineer",
    "sales executive",
    "customer support",
    "hr manager",
    "graphic designer",
    "marketing manager",
    "sales manager",
    "operations manager",
    "mechanical engineer",
}

STRONG_AI_TITLES = {
    "ai engineer",
    "machine learning engineer",
    "ml engineer",
    "nlp engineer",
    "search engineer",
    "recommendation systems engineer",
    "recommendation engineer",
    "recommender systems engineer",
    "retrieval engineer",
    "ranking engineer",
}

DOMAIN_TERMS = {
    "retrieval",
    "information retrieval",
    "ranking",
    "learning to rank",
    "recommendation",
    "recommendation systems",
    "semantic search",
    "vector search",
    "vector database",
    "embeddings",
    "pinecone",
    "faiss",
    "milvus",
    "qdrant",
    "weaviate",
    "rag",
    "langchain",
    "llamaindex",
    "search",
    "nlp",
    "machine learning",
    "deep learning",
    "ai",
    "ml",
}

NEGATIVE_ROLE_TERMS = {
    "civil engineer",
    "sales",
    "support",
    "customer support",
    "hr",
    "designer",
    "marketing",
    "operations manager",
    "mechanical engineer",
}


@dataclass
class CandidateDiagnostic:
    candidate_id: str
    rank: int
    current_title: str
    final_score: float
    semantic_score: float
    skill_score: float
    experience_score: float
    behavior_score: float
    product_fit_score: float
    title_score: float
    is_irrelevant: bool
    reasons: List[str]
    fix_suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize(text: Any) -> str:
    return str(text or "").strip().lower()


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def is_obviously_irrelevant(row: Dict[str, Any]) -> bool:
    title = _normalize(row.get("current_title"))
    skills = " ".join(map(_normalize, row.get("skills") or []))
    reasoning = _normalize(row.get("reasoning"))
    career = _normalize(row.get("career_history_text", ""))
    blob = f"{title} {skills} {reasoning} {career}"

    if _contains_any(blob, REJECT_TITLES):
        return True

    if row.get("title_score", 0) <= 0.2 and row.get("skill_score", 0) < 0.25 and row.get("product_fit_score", 0) < 0.2:
        return True

    if row.get("semantic_score", 0) > 0.55 and row.get("title_score", 0) <= 0.25 and row.get("product_fit_score", 0) < 0.2:
        return True

    return False


def explain_high_rank(row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    title = _normalize(row.get("current_title"))
    skills = " ".join(map(_normalize, row.get("skills") or []))
    career = _normalize(row.get("career_history_text", ""))
    domain_blob = f"{title} {skills} {career}"

    if row.get("title_score", 0) >= 0.75:
        reasons.append("Strong title relevance to the target AI/search role.")
    elif row.get("title_score", 0) >= 0.55:
        reasons.append("Moderate title relevance with some career-path alignment.")
    elif row.get("semantic_score", 0) >= 0.5:
        reasons.append("Rank boosted mainly by generic semantic overlap with the job description.")

    domain_hits = [term for term in DOMAIN_TERMS if term in domain_blob]
    if domain_hits:
        reasons.append(f"Domain keywords matched: {', '.join(domain_hits[:5])}.")

    if row.get("skill_score", 0) >= 0.35:
        reasons.append("AI/ML skill coverage is materially above average.")

    if row.get("product_fit_score", 0) >= 0.35:
        reasons.append("Product-fit terms suggest search, retrieval, ranking, or recommendation experience.")

    if row.get("behavior_score", 0) >= 0.6:
        reasons.append("Behavioral signals are healthy enough to support the rank.")

    if not reasons:
        reasons.append("Candidate reached the shortlist through a mixed signal profile rather than one dominant signal.")

    return reasons


def explain_false_positive(row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    title = _normalize(row.get("current_title"))
    skills = " ".join(map(_normalize, row.get("skills") or []))
    career = _normalize(row.get("career_history_text", ""))
    blob = f"{title} {skills} {career}"

    if _contains_any(blob, REJECT_TITLES):
        reasons.append("Current title or history contains a clearly unrelated role for an AI Engineer search.")

    if row.get("semantic_score", 0) >= 0.5:
        reasons.append("Generic semantic similarity likely overpowered more specific career signals.")

    if row.get("title_score", 0) <= 0.3:
        reasons.append("Title relevance was too weak to filter the profile out.")

    if row.get("skill_score", 0) <= 0.25:
        reasons.append("AI/ML skill evidence was insufficient or too generic.")

    if row.get("product_fit_score", 0) <= 0.2:
        reasons.append("Product-fit keywords did not provide enough domain alignment.")

    if not reasons:
        reasons.append("The candidate survived ranking despite low domain alignment signals.")

    return reasons


def suggest_scoring_fixes(top_rows: Sequence[Dict[str, Any]]) -> List[str]:
    total = len(top_rows)
    if total == 0:
        return ["No ranked candidates were available for diagnostics."]

    false_positive_rows = [row for row in top_rows if is_obviously_irrelevant(row)]
    false_positive_count = len(false_positive_rows)
    semantic_leaks = [
        row for row in top_rows
        if row.get("semantic_score", 0) >= 0.5 and row.get("title_score", 0) <= 0.35 and row.get("product_fit_score", 0) < 0.25
    ]
    weak_domain = [
        row for row in top_rows
        if row.get("skill_score", 0) < 0.25 and row.get("product_fit_score", 0) < 0.25
    ]

    suggestions: List[str] = []

    if false_positive_count > 0:
        suggestions.append(
            f"Increase the title score weight and harden reject titles because {false_positive_count}/{total} top candidates are clearly irrelevant."
        )

    if semantic_leaks:
        suggestions.append(
            "Reduce semantic similarity influence further or apply stronger calibration so buzzword overlap does not outrank career relevance."
        )

    if weak_domain:
        suggestions.append(
            "Boost product-fit and domain-skill weights for retrieval, ranking, search, embeddings, and recommendation keywords."
        )

    if any(row.get("title_score", 0) <= 0.25 for row in top_rows[:10]):
        suggestions.append(
            "Penalize generic titles such as Software Engineer unless career history contains AI/search/retrieval signals."
        )

    if not suggestions:
        suggestions.append(
            "Ranking quality looks acceptable in the shortlist, but keep a small penalty on generic semantic overlap."
        )

    return suggestions


def diagnose_top_candidates(top_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    diagnostics: List[CandidateDiagnostic] = []
    irrelevant_count = 0
    for row in top_rows:
        irrelevant = is_obviously_irrelevant(row)
        if irrelevant:
            irrelevant_count += 1
            reasons = explain_false_positive(row)
        else:
            reasons = explain_high_rank(row)

        diagnostics.append(
            CandidateDiagnostic(
                candidate_id=str(row.get("candidate_id", "")),
                rank=int(row.get("rank", 0) or 0),
                current_title=str(row.get("current_title", "")),
                final_score=float(row.get("final_score", 0) or 0),
                semantic_score=float(row.get("semantic_score", 0) or 0),
                skill_score=float(row.get("skill_score", 0) or 0),
                experience_score=float(row.get("experience_score", 0) or 0),
                behavior_score=float(row.get("behavior_score", 0) or 0),
                product_fit_score=float(row.get("product_fit_score", 0) or 0),
                title_score=float(row.get("title_score", 0) or 0),
                is_irrelevant=irrelevant,
                reasons=reasons,
                fix_suggestions=[],
            )
        )

    shared_fixes = suggest_scoring_fixes(top_rows)
    for diag in diagnostics:
        diag.fix_suggestions = shared_fixes[:]

    return {
        "top_count": len(top_rows),
        "irrelevant_count": irrelevant_count,
        "false_positive_rate": round((irrelevant_count / len(top_rows)), 4) if top_rows else 0.0,
        "candidate_diagnostics": [diag.to_dict() for diag in diagnostics],
        "suggested_fixes": shared_fixes,
    }
