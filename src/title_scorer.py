class TitleScorer:

    def __init__(self):
        self.strong_titles = [
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
        ]

        self.medium_titles = [
            "backend engineer",
            "data engineer",
            "ai specialist",
            "applied scientist",
            "research engineer",
        ]

        self.reject_titles = [
            "civil engineer",
            "sales executive",
            "sales manager",
            "customer support",
            "hr manager",
            "graphic designer",
            "marketing manager",
            "operations manager",
            "mechanical engineer",
        ]

        self.career_positive_terms = [
            "ai engineer",
            "machine learning engineer",
            "ml engineer",
            "nlp engineer",
            "search engineer",
            "recommendation systems engineer",
            "retrieval",
            "ranking",
            "recommendation systems",
            "semantic search",
            "vector search",
            "information retrieval",
            "embeddings",
            "pinecone",
            "faiss",
            "milvus",
            "qdrant",
            "weaviate",
            "rag",
            "langchain",
            "llamaindex",
            "machine learning",
            "deep learning",
            "nlp",
            "llm",
            "llms",
            "search",
            "ranking",
        ]

        self.career_negative_terms = [
            "civil engineer",
            "sales executive",
            "sales manager",
            "customer support",
            "hr manager",
            "graphic designer",
            "marketing manager",
            "operations manager",
            "mechanical engineer",
        ]

    def calculate(self, title, career_history=None):
        title_text, history_text = self._normalize_inputs(title, career_history)

        if not title_text and not history_text:
            return 0.0

        if self._contains_any(title_text, self.reject_titles):
            return -1.0

        if self._contains_any(title_text, self.strong_titles):
            base_score = 1.0
        elif "software engineer" in title_text:
            base_score = 0.25
        elif self._contains_any(title_text, self.medium_titles):
            base_score = 0.6
        else:
            base_score = 0.1

        career_positive_hits = self._count_hits(history_text, self.career_positive_terms)
        career_negative_hits = self._count_hits(history_text, self.career_negative_terms)
        history_bonus = min(0.35, career_positive_hits * 0.08)
        history_penalty = min(0.6, career_negative_hits * 0.18)

        score = base_score + history_bonus - history_penalty

        if self._contains_any(title_text, ["backend engineer", "data engineer"]) and career_positive_hits >= 2:
            score = max(score, 0.82)
        elif self._contains_any(title_text, ["backend engineer", "data engineer"]) and career_positive_hits >= 1:
            score = max(score, 0.7)

        if "software engineer" in title_text and career_positive_hits >= 2:
            score = max(score, 0.74)
        elif "software engineer" in title_text and career_positive_hits >= 1:
            score = max(score, 0.55)

        if self._contains_any(title_text, ["ai specialist", "applied scientist", "research engineer"]) and career_positive_hits >= 2:
            score = max(score, 0.86)

        if career_negative_hits >= 2 and career_positive_hits == 0:
            score = min(score, -0.5)

        return round(max(-1.0, min(score, 1.0)), 4)

    def _normalize_inputs(self, title, career_history):
        if hasattr(title, "current_title"):
            candidate = title
            title = getattr(candidate, "current_title", "")
            career_history = getattr(candidate, "career_history", career_history)

        title_text = (title or "").strip().lower()
        history_text = self._history_text(career_history)
        return title_text, history_text

    def _history_text(self, career_history):
        parts = []
        for job in career_history or []:
            if not isinstance(job, dict):
                continue
            title = (job.get("title") or "").strip()
            description = (job.get("description") or "").strip()
            if title:
                parts.append(title)
            if description:
                parts.append(description)
        return " ".join(parts).lower()

    def _contains_any(self, text, phrases):
        text = text or ""
        return any(phrase in text for phrase in phrases)

    def _count_hits(self, text, phrases):
        text = text or ""
        return sum(1 for phrase in phrases if phrase in text)
