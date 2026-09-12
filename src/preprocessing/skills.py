import re


# Canonical skill names and their possible aliases
SKILL_ALIASES = {
    "python": [
        "python"
    ],
    "java": [
        "java"
    ],
    "sql": [
        "sql"
    ],
    "pandas": [
        "pandas"
    ],
    "numpy": [
        "numpy"
    ],
    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],
    "tensorflow": [
        "tensorflow"
    ],
    "nlp": [
        "nlp",
        "natural language processing"
    ],
    "machine learning": [
        "machine learning",
        "machine-learning",
        "ml"
    ],
    "spring boot": [
        "spring boot"
    ],
    "mysql": [
        "mysql"
    ],
    "postgresql": [
        "postgresql",
        "postgres"
    ],
    "rest api": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis"
    ],
    "django": [
        "django"
    ],
    "javascript": [
        "javascript"
    ],
    "react": [
        "react",
        "react.js"
    ],
    "typescript": [
        "typescript"
    ],
    "html": [
        "html"
    ],
    "css": [
        "css"
    ],
    "excel": [
        "excel",
        "microsoft excel"
    ],
    "git": [
        "git"
    ],
    "docker": [
        "docker"
    ],
    "microservices": [
        "microservices",
        "microservice"
    ]
}

# Skill importance weights
SKILL_WEIGHTS = {
    "python": 3,
    "java": 3,
    "javascript": 3,

    "machine learning": 3,
    "scikit-learn": 3,
    "tensorflow": 3,
    "nlp": 3,

    "django": 3,
    "spring boot": 3,
    "react": 3,

    "sql": 2,
    "pandas": 2,
    "numpy": 2,
    "mysql": 2,
    "postgresql": 2,
    "rest api": 2,
    "typescript": 2,

    "html": 1,
    "css": 1,
    "excel": 1,
    "git": 1,
    "docker": 1,
    "microservices": 1
}

def extract_skills(text):
    """
    Extract canonical skills from text.

    Different aliases are normalized to one
    canonical skill name.
    """

    text = text.lower()

    found_skills = []

    for canonical_skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            pattern = r"\b" + re.escape(alias) + r"\b"

            if re.search(pattern, text):

                found_skills.append(canonical_skill)

                # Stop checking aliases once one matches
                break

    return found_skills

def calculate_weighted_skill_match(
    required_skills,
    candidate_skills
):
    """
    Calculate weighted skill match ratio.

    More important skills contribute more
    to the final score.
    """

    if not required_skills:
        return 0

    required_skills = set(required_skills)
    candidate_skills = set(candidate_skills)

    matched_skills = (
        required_skills
        & candidate_skills
    )

    total_weight = sum(
        SKILL_WEIGHTS.get(skill, 1)
        for skill in required_skills
    )

    matched_weight = sum(
        SKILL_WEIGHTS.get(skill, 1)
        for skill in matched_skills
    )

    return matched_weight / total_weight