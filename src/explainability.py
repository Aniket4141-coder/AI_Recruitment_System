from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple


class ExplainabilityGenerator:
    def __init__(self, skill_matcher=None):
        self.skill_matcher = skill_matcher
        self.stop_words = {
            "and",
            "or",
            "the",
            "a",
            "an",
            "to",
            "of",
            "for",
            "in",
            "on",
            "with",
            "by",
            "from",
            "is",
            "are",
            "as",
            "at",
            "be",
            "this",
            "that",
            "these",
            "those",
            "we",
            "you",
            "your",
            "our",
            "their",
            "role",
            "job",
            "candidate",
            "experience",
            "skills",
            "skill",
        }

    def generate(
        self,
        candidate,
        jd_text: str,
        semantic_score: float,
        skill_score: float,
        behavior_score: float,
        experience_score: float,
        product_fit_score: float,
        title_score: float,
    ) -> Dict[str, str]:
        selected = self._build_selected_reasons(
            candidate=candidate,
            jd_text=jd_text,
            semantic_score=semantic_score,
            skill_score=skill_score,
            behavior_score=behavior_score,
            experience_score=experience_score,
            product_fit_score=product_fit_score,
            title_score=title_score,
        )
        not_selected = self._build_gap_reasons(
            candidate=candidate,
            jd_text=jd_text,
            skill_score=skill_score,
            behavior_score=behavior_score,
            experience_score=experience_score,
            semantic_score=semantic_score,
        )

        return {
            "why_selected": self._format_bullets(selected, fallback="This profile aligns well with the role requirements."),
            "why_not_selected": self._format_bullets(
                not_selected,
                fallback="No major gaps were detected from the available signals.",
            ),
            "explanation": self._format_explanation(selected, not_selected),
        }

    def _build_selected_reasons(
        self,
        candidate,
        jd_text: str,
        semantic_score: float,
        skill_score: float,
        behavior_score: float,
        experience_score: float,
        product_fit_score: float,
        title_score: float,
    ) -> List[str]:
        jd_terms = self._extract_keywords(jd_text)
        matched_skills = self._matching_skills(candidate.skills, jd_terms)
        career_matches = self._matching_career_history(candidate.career_history, jd_terms)

        reasons: List[str] = []

        if matched_skills:
            reasons.append(
                f"Matching skills: {self._join_items(matched_skills[:6])} are directly relevant to the role."
            )
        elif skill_score > 0.6:
            reasons.append(
                "Matching skills: the profile shows a strong general skill fit for this role."
            )

        experience_text = self._experience_reason(candidate.experience, experience_score)
        if experience_text:
            reasons.append(experience_text)

        if career_matches:
            reasons.append(
                f"Matching career history: prior roles such as {self._join_items(career_matches[:3])} align with the job theme."
            )
        elif title_score >= 0.6:
            reasons.append(
                f"Matching career history: the current title '{candidate.current_title}' is a close match for the target role."
            )

        semantic_text = self._semantic_reason(semantic_score)
        if semantic_text:
            reasons.append(semantic_text)

        if product_fit_score >= 0.5:
            reasons.append(
                "Product fit: the resume signals domain alignment with search, ranking, recommendation, or adjacent AI work."
            )

        if behavior_score >= 0.6:
            reasons.append(
                "Engagement: strong platform activity and responsiveness suggest this candidate is likely recruitable."
            )

        return reasons

    def _build_gap_reasons(
        self,
        candidate,
        jd_text: str,
        skill_score: float,
        behavior_score: float,
        experience_score: float,
        semantic_score: float,
    ) -> List[str]:
        jd_terms = self._extract_keywords(jd_text)
        matched_skills = {self._normalize(skill) for skill in self._matching_skills(candidate.skills, jd_terms)}
        required_skills = self._priority_skills()
        missing_skills = [
            skill
            for skill in required_skills
            if self._normalize(skill) not in matched_skills
        ]

        reasons: List[str] = []

        if missing_skills and skill_score < 0.85:
            reasons.append(
                f"Missing skills: no strong evidence for {self._join_items(missing_skills[:5])} in the current profile."
            )

        if experience_score < 0.8:
            reasons.append(
                "Weak experience: the candidate appears either early in career, outside the preferred range, or not deeply aligned with the role scope."
            )

        low_engagement_signals = self._engagement_gaps(candidate.redrob_signals, behavior_score)
        if low_engagement_signals:
            reasons.append(
                f"Low engagement: {self._join_items(low_engagement_signals)}."
            )

        if semantic_score < 0.4:
            reasons.append(
                "Semantic relevance is limited: the overall profile language does not strongly mirror the job description."
            )

        return reasons

    def _experience_reason(self, years: float, score: float) -> str:
        if score >= 0.95:
            return f"Relevant experience: {years:.1f} years is strongly aligned with the expected experience band."
        if score >= 0.75:
            return f"Relevant experience: {years:.1f} years is a good fit for this opening."
        if score >= 0.6:
            return f"Relevant experience: {years:.1f} years is broadly relevant, though not a perfect match."
        return ""

    def _semantic_reason(self, semantic_score: float) -> str:
        if semantic_score >= 0.6:
            return "Semantic relevance: the profile language and job description show strong topical overlap."
        if semantic_score >= 0.4:
            return "Semantic relevance: there is moderate overlap between the candidate story and the role."
        if semantic_score > 0:
            return "Semantic relevance: only limited overlap is visible between the profile and the job description."
        return ""

    def _engagement_gaps(self, signals: Dict[str, Any], behavior_score: float) -> List[str]:
        gaps: List[str] = []

        completeness = float(signals.get("profile_completeness_score", 0) or 0)
        response_rate = float(signals.get("recruiter_response_rate", 0) or 0)
        interview_completion = float(signals.get("interview_completion_rate", 0) or 0)
        github_activity = float(signals.get("github_activity_score", 0) or 0)

        if behavior_score < 0.5 or completeness < 60:
            gaps.append("profile completeness looks light")
        if response_rate < 0.25:
            gaps.append("recruiter response rate is low")
        if interview_completion < 0.5:
            gaps.append("interview completion looks weak")
        if github_activity <= 0:
            gaps.append("GitHub activity is absent or uninformative")

        return gaps[:3]

    def _matching_skills(self, candidate_skills: Sequence[str], jd_terms: Iterable[str]) -> List[str]:
        normalized_terms = {self._normalize(term) for term in jd_terms}
        matches: List[str] = []

        for skill in candidate_skills:
            normalized_skill = self._normalize(skill)
            if not normalized_skill:
                continue
            if normalized_skill in normalized_terms:
                matches.append(skill)
                continue

            if any(term in normalized_skill or normalized_skill in term for term in normalized_terms):
                matches.append(skill)

        if self.skill_matcher is not None:
            weighted_hits = []
            for skill in getattr(self.skill_matcher, "weighted_skills", {}).keys():
                normalized_skill = self._normalize(skill)
                if any(normalized_skill in self._normalize(cand_skill) for cand_skill in candidate_skills):
                    weighted_hits.append(skill)
            for hit in weighted_hits:
                if hit not in matches:
                    matches.append(hit)

        return matches

    def _matching_career_history(self, career_history: Sequence[Dict[str, Any]], jd_terms: Iterable[str]) -> List[str]:
        normalized_terms = {self._normalize(term) for term in jd_terms}
        matches: List[str] = []

        for job in career_history or []:
            if not isinstance(job, dict):
                continue
            title = (job.get("title") or "").strip()
            description = (job.get("description") or "").strip()
            haystack = f"{title} {description}".lower()
            if any(term in haystack for term in normalized_terms):
                if title:
                    matches.append(title)
                elif description:
                    matches.append(description[:60].rstrip())

        return matches

    def _priority_skills(self) -> List[str]:
        if self.skill_matcher is None:
            return [
                "retrieval",
                "ranking",
                "recommendation systems",
                "semantic search",
                "vector search",
                "embeddings",
                "pinecone",
                "faiss",
                "milvus",
                "qdrant",
                "weaviate",
                "rag",
                "sentence transformers",
                "llm",
                "machine learning",
                "python",
            ]
        return list(getattr(self.skill_matcher, "weighted_skills", {}).keys())

    def _extract_keywords(self, text: str, limit: int = 30) -> List[str]:
        text = (text or "").lower()
        candidates = re.findall(r"[a-z][a-z0-9+\-#/ ]{2,}", text)
        counts = Counter()

        for candidate in candidates:
            token = candidate.strip()
            if not token:
                continue
            if token in self.stop_words:
                continue
            if len(token) < 3:
                continue
            counts[token] += 1

        return [item for item, _ in counts.most_common(limit)]

    def _format_bullets(self, reasons: Sequence[str], fallback: str) -> str:
        if not reasons:
            return fallback
        return "\n".join(f"- {reason}" for reason in reasons)

    def _format_explanation(self, selected: Sequence[str], not_selected: Sequence[str]) -> str:
        selected_text = "; ".join(selected) if selected else "Strong overall alignment with the role."
        gap_text = "; ".join(not_selected) if not_selected else "No material gaps were detected from the available signals."
        return f"Why selected: {selected_text} | Why not selected: {gap_text}"

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _join_items(self, items: Sequence[str]) -> str:
        cleaned = [item for item in items if item]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"
