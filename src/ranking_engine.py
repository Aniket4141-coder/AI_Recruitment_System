class RankingEngine:

    def calculate_final_score(
        self,
        semantic_score,
        skill_score,
        experience_score,
        behavior_score,
        product_fit_score,
        title_score,
    ):
        # Hackathon-friendly weighting: prioritize title, skill, and domain fit over generic semantic overlap.
        final_score = (
            0.10 * semantic_score +
            0.25 * skill_score +
            0.07 * experience_score +
            0.08 * behavior_score +
            0.20 * product_fit_score +
            0.30 * title_score
        )

        return round(max(0.0, final_score), 4)
