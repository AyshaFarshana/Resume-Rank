from src.preprocessing.skills import (
    extract_skills,
    calculate_weighted_skill_match,
)


def test_extract_skills():
    text = """
    Python developer with experience in Django,
    PostgreSQL, REST API development and Git.
    """

    skills = extract_skills(text)

    assert "python" in skills
    assert "django" in skills
    assert "postgresql" in skills
    assert "rest api" in skills
    assert "git" in skills


def test_weighted_skill_match():
    required = [
        "python",
        "django",
        "postgresql",
        "rest api",
        "git",
    ]

    candidate = [
        "python",
        "django",
        "postgresql",
        "rest api",
        "git",
    ]

    score = calculate_weighted_skill_match(
        required,
        candidate,
    )

    assert score == 1.0