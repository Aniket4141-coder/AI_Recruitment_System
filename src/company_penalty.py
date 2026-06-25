class RankingEngine:

    def calculate_final_score(

        self,

        semantic_score,

        skill_score,

        experience_score,

        behavior_score,

        product_fit_score,

        company_penalty

    ):

        final_score = (

            0.35 * semantic_score +

            0.25 * skill_score +

            0.10 * experience_score +

            0.15 * behavior_score +

            0.15 * product_fit_score +

            company_penalty

        )

        return round(
            max(0, final_score),
            4
        )