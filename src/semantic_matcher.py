from __future__ import annotations

import logging
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


LOGGER = logging.getLogger(__name__)


class SemanticMatcher:

    def __init__(self):
        self.last_jd_text = None
        self.last_jd_embedding = None
        self.use_fallback = False
        self.model = None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        if SentenceTransformer is None:
            self.use_fallback = True
            LOGGER.warning("sentence-transformers is unavailable; using TF-IDF fallback for semantic scoring.")
            return

        if not self._has_local_model_cache(self.model_name):
            self.use_fallback = True
            LOGGER.warning(
                "No local cache found for %s; using TF-IDF fallback for semantic scoring.",
                self.model_name,
            )
            return

        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception as exc:
            self.use_fallback = True
            self.model = None
            LOGGER.warning(
                "Falling back to TF-IDF semantic scoring because the transformer model could not be loaded: %s",
                exc,
            )

    def _cache_roots(self):
        roots = []
        env_vars = [
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_CACHE",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
        ]
        for env_name in env_vars:
            value = os.getenv(env_name)
            if value:
                roots.append(Path(value))

        roots.extend(
            [
                Path.home() / ".cache" / "huggingface" / "hub",
                Path.home() / ".cache" / "huggingface",
            ]
        )
        return roots

    def _has_local_model_cache(self, model_name):
        cache_dir_name = "models--" + model_name.replace("/", "--")

        for root in self._cache_roots():
            candidate_root = root / cache_dir_name
            snapshots = candidate_root / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                return True

        return False

    def _calibrate_similarity(self, value):
        value = max(0.0, min(1.0, float(value)))
        # Compress mid-range scores so generic overlap does not dominate ranking.
        calibrated = value ** 1.2
        return round(calibrated, 4)

    def _calculate_similarity_fallback(self, jd_text, candidate_texts):
        if not candidate_texts:
            return []

        corpus = [jd_text] + list(candidate_texts)
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        tfidf = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:])
        return [self._calibrate_similarity(float(val)) for val in similarities[0]]

    def calculate_similarity(self, jd_text, candidate_text):
        if self.use_fallback or self.model is None:
            return self._calculate_similarity_fallback(jd_text, [candidate_text])[0]

        if self.last_jd_text == jd_text:
            jd_embedding = self.last_jd_embedding
        else:
            jd_embedding = self.model.encode([jd_text])
            self.last_jd_text = jd_text
            self.last_jd_embedding = jd_embedding

        candidate_embedding = self.model.encode([candidate_text])
        similarity = cosine_similarity(jd_embedding, candidate_embedding)
        return self._calibrate_similarity(float(similarity[0][0]))

    def calculate_similarity_batch(self, jd_text, candidate_texts):
        if not candidate_texts:
            return []

        if self.use_fallback or self.model is None:
            return self._calculate_similarity_fallback(jd_text, candidate_texts)

        if self.last_jd_text == jd_text:
            jd_embedding = self.last_jd_embedding
        else:
            jd_embedding = self.model.encode([jd_text])
            self.last_jd_text = jd_text
            self.last_jd_embedding = jd_embedding

        candidate_embeddings = self.model.encode(
            candidate_texts,
            batch_size=64,
            show_progress_bar=False,
        )

        similarities = cosine_similarity(jd_embedding, candidate_embeddings)
        return [self._calibrate_similarity(float(val)) for val in similarities[0]]
