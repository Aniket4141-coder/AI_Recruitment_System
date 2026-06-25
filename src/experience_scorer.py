class ExperienceScorer:

    def calculate(
        self,
        years
    ):

        if 5 <= years <= 9:
            return 1.0

        elif 3 <= years < 5:
            return 0.8

        elif 9 < years <= 12:
            return 0.7

        elif years > 12:
            return 0.6

        else:
            return 0.4