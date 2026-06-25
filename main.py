import json

from src.parser import Candidate

from src.jd_processor import load_job_description

from src.semantic_matcher import SemanticMatcher

from src.skill_matcher import SkillMatcher

from src.experience_scorer import ExperienceScorer

from src.behavioral_scorer import BehavioralScorer

from src.product_fit import ProductFitScorer

from src.ranking_engine import RankingEngine

from src.reasoning_generator import ReasoningGenerator

from src.title_scorer import TitleScorer


jd = load_job_description(
    "data/job_description.docx"
)

with open(
    "data/candidates.jsonl",
    "r",
    encoding="utf-8"
) as f:

    candidates = [
        json.loads(line)
        for line in f
    ]

semantic_matcher = SemanticMatcher()

skill_matcher = SkillMatcher()

experience_scorer = ExperienceScorer()

behavioral_scorer = BehavioralScorer()

product_fit_scorer = ProductFitScorer()

ranking_engine = RankingEngine()

reasoning_generator = ReasoningGenerator()

title_scorer = TitleScorer()

test_candidates = candidates[:100]

results = []

for data in test_candidates:

    candidate = Candidate(data)

    candidate_text = candidate.get_candidate_text()

    semantic_score = (
        semantic_matcher.calculate_similarity(
            jd,
            candidate_text
        )
    )

    skill_score = (
        skill_matcher.calculate_skill_score(
            candidate.skills
        )
    )

    experience_score = (
        experience_scorer.calculate(
            candidate.experience
        )
    )

    behavior_score = (
        behavioral_scorer.calculate(
            candidate.redrob_signals
        )
    )

    product_fit_score = (
    product_fit_scorer.calculate(
        candidate
    )
)
    
    title_score = (
    title_scorer.calculate(candidate.current_title, candidate.career_history)
)
    
    print(
    candidate.current_title,
    round(semantic_score, 4),
    round(skill_score, 4),
    round(product_fit_score, 4),
    round(title_score, 4)
    )


    final_score = (
    ranking_engine.calculate_final_score(
        semantic_score,
        skill_score,
        experience_score,
        behavior_score,
        product_fit_score,
        title_score
    )
)

    reasoning = (
        reasoning_generator.generate(
            candidate,
            final_score
        )
    )

    results.append({

        "candidate_id": candidate.id,

        "score": final_score,

        "reasoning": reasoning

    })

    results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)
    
for rank, row in enumerate(
    results,
    start=1
):

    row["rank"] = rank

print("\nTOP 10 CANDIDATES\n")

print("\nGRAPHIC DESIGNER ANALYSIS\n")

print("\nSKILLS:")
print(candidate.skills)

for data in candidates:

    if data["candidate_id"] == "CAND_0000083":

        candidate = Candidate(data)

        candidate_text = candidate.get_candidate_text()

        semantic_score = semantic_matcher.calculate_similarity(
            jd,
            candidate_text
        )

        skill_score = skill_matcher.calculate_skill_score(
            candidate.skills
        )

        experience_score = experience_scorer.calculate(
            candidate.experience
        )

        behavior_score = behavioral_scorer.calculate(
            candidate.redrob_signals
        )

        product_fit_score = product_fit_scorer.calculate(
            candidate
        )

        title_score = title_scorer.calculate(candidate.current_title, candidate.career_history)

        print("Title:", candidate.current_title)

        print("Semantic:", semantic_score)

        print("Skill:", skill_score)

        print("Experience:", experience_score)

        print("Behavior:", behavior_score)

        print("Product Fit:", product_fit_score)

        print("Title Score:", title_score)

        print("Skills:", candidate.skills)

        break

for row in results[:10]:

    print(
        row["rank"],
        row["candidate_id"],
        row["score"]
    )    


print("\nTOP 5 CANDIDATE DETAILS\n")

top_ids = [
    row["candidate_id"]
    for row in results[:5]
]

for candidate_id in top_ids:

    for data in candidates:

        if data["candidate_id"] == candidate_id:

            profile = data["profile"]

            print("=" * 50)

            print("ID:", candidate_id)

            print(
                "Title:",
                profile["current_title"]
            )

            print(
                "Experience:",
                profile["years_of_experience"]
            )

            print(
                "Company:",
                profile["current_company"]
            )

            print(
                "Headline:",
                profile["headline"]
            )

            print()

            break    

print("\nRECOMMENDATION ENGINEER ANALYSIS\n")

print("\nSKILLS:")
print(candidate.skills)

for data in candidates:

    if data["candidate_id"] == "CAND_0000031":

        candidate = Candidate(data)

        candidate_text = candidate.get_candidate_text()

        semantic_score = semantic_matcher.calculate_similarity(
            jd,
            candidate_text
        )

        skill_score = skill_matcher.calculate_skill_score(
            candidate.skills
        )

        experience_score = experience_scorer.calculate(
            candidate.experience
        )

        behavior_score = behavioral_scorer.calculate(
            candidate.redrob_signals
        )

        product_fit_score = product_fit_scorer.calculate(
            candidate
        )

        title_score = title_scorer.calculate(candidate.current_title, candidate.career_history)

        print("Title:", candidate.current_title)

        print("Semantic:", semantic_score)

        print("Skill:", skill_score)

        print("Experience:", experience_score)

        print("Behavior:", behavior_score)

        print("Product Fit:", product_fit_score)

        print("Title Score:", title_score)

        print("Skills:", candidate.skills)

        break        