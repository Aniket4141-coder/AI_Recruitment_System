class ReasoningGenerator:

    def generate(
        self,
        candidate,
        score
    ):

        return (

            f"{candidate.current_title} with "

            f"{candidate.experience} years experience; "

            f"{len(candidate.skills)} skills listed; "

            f"final score {score}."

        )