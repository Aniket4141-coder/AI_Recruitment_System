class SkillMatcher:

    def __init__(self):
        # Domain-first weighting. Generic keywords are intentionally light so they do not
        # overpower search / retrieval / ranking / AI relevance.
        self.weighted_skills = {
            "retrieval": 14,
            "information retrieval": 14,
            "learning to rank": 14,
            "ranking": 14,
            "recommendation systems": 13,
            "recommendation": 12,
            "semantic search": 12,
            "vector search": 12,
            "search": 10,
            "semantic similarity": 10,
            "vector database": 10,
            "embeddings": 10,
            "pinecone": 9,
            "faiss": 9,
            "milvus": 9,
            "qdrant": 9,
            "weaviate": 9,
            "rag": 8,
            "langchain": 8,
            "llamaindex": 8,
            "machine learning": 8,
            "deep learning": 8,
            "nlp": 8,
            "ai": 8,
            "sentence transformers": 7,
            "transformers": 6,
            "llm": 6,
            "llms": 6,
            "elasticsearch": 8,
            "opensearch": 8,
            "pytorch": 5,
            "tensorflow": 5,
            "fastapi": 3,
            "spark": 3,
            "airflow": 3,
            "kafka": 3,
            "python": 2,
        }

    def calculate_skill_score(self, candidate_skills):
        import re

        candidate_skills = candidate_skills or []
        if not candidate_skills:
            return 0.0

        score = 0
        max_score = sum(self.weighted_skills.values())

        for skill, weight in self.weighted_skills.items():
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            for cand_skill in candidate_skills:
                if re.search(pattern, (cand_skill or '').lower()):
                    score += weight
                    break

        return round(score / max_score, 4)
