class ProductFitScorer:

    def __init__(self):
        self.positive_keywords = {
            "recommendation systems": 4,
            "information retrieval": 4,
            "semantic search": 4,
            "vector search": 4,
            "learning to rank": 4,
            "retrieval": 4,
            "ranking": 4,
            "search": 3,
            "semantic similarity": 3,
            "vector database": 3,
            "embeddings": 3,
            "pinecone": 3,
            "faiss": 3,
            "milvus": 3,
            "qdrant": 3,
            "weaviate": 3,
            "rag": 4,
            "langchain": 3,
            "llamaindex": 3,
            "ai engineer": 2,
            "ml engineer": 2,
            "machine learning engineer": 2,
            "nlp engineer": 2,
            "search engineer": 2,
            "recommendation engineer": 2,
            "data engineer": 1,
            "backend engineer": 1,
        }

        self.negative_keywords = {
            "civil engineer": 5,
            "sales executive": 5,
            "customer support": 5,
            "hr manager": 5,
            "graphic designer": 5,
            "marketing manager": 5,
            "sales manager": 4,
            "operations manager": 4,
            "sales": 3,
            "marketing": 3,
            "support": 2,
            "hr": 2,
            "designer": 2,
            "civil": 2,
            "human resources": 3,
            "accounting": 2,
            "accountant": 2,
        }

    def calculate(self, candidate):
        import re

        text = (
            (candidate.summary or "") + " " +
            (candidate.current_title or "") + " " +
            candidate.get_candidate_text()
        ).lower()

        positive_points = 0
        negative_points = 0

        for keyword, weight in self.positive_keywords.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                positive_points += weight

        for keyword, weight in self.negative_keywords.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                negative_points += weight

        # Reward strong AI/search/retrieval alignment and apply stronger penalties for unrelated domains.
        raw_score = positive_points - (1.35 * negative_points)

        if positive_points == 0 and negative_points == 0:
            return 0.0

        max_positive = sum(self.positive_keywords.values())
        normalized = raw_score / max_positive
        return round(max(0.0, min(normalized, 1.0)), 4)
