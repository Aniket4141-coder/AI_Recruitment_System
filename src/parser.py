class Candidate:

    def __init__(self, data):

        data = data or {}

        self.id = data.get("candidate_id") or ""

        profile = data.get("profile") or {}

        self.name = profile.get("anonymized_name") or ""
        self.current_title = profile.get("current_title") or ""
        self.current_company = profile.get("current_company") or ""
        self.experience = profile.get("years_of_experience") or 0
        self.summary = profile.get("summary") or ""
        self.location = profile.get("location") or ""
        self.country = profile.get("country") or ""

        self.skills = [
            skill.get("name", "")
            for skill in (data.get("skills") or [])
            if isinstance(skill, dict)
        ]

        self.career_history = data.get("career_history") or []

        self.redrob_signals = data.get(
            "redrob_signals", {}
        ) or {}

    def get_career_text(self):

        text = ""

        for job in self.career_history:

            if not isinstance(job, dict):
                continue

            text += " "
            text += job.get("title", "")
            text += " "
            text += job.get("description", "")

        return text

    def get_candidate_text(self):

        skills_text = " ".join(self.skills)

        career_text = self.get_career_text()

        return f"""
        Current Title:
        {self.current_title}

        Summary:
        {self.summary}

        Career:
        {career_text}

        Skills:
        {skills_text}
        """