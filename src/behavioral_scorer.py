class BehavioralScorer:

    def calculate(self, signals):

        profile = min(
            signals.get(
                "profile_completeness_score",
                0
            ) / 100,
            1
        )

        response = min(
            signals.get(
                "recruiter_response_rate",
                0
            ),
            1
        )

        interview = min(
            signals.get(
                "interview_completion_rate",
                0
            ),
            1
        )

        github_score = max(
            0,
            signals.get(
                "github_activity_score",
                0
            )
        )

        github = min(
            github_score / 100,
            1
        )

        return round(
            (
                profile +
                response +
                interview +
                github
            ) / 4,
            4
        )