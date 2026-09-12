import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.skills import (
    calculate_weighted_skill_match,
    extract_skills,
)


# ============================================================
# 1. Load data
# ============================================================

resumes = pd.read_csv("data/sample/resumes.csv")
jobs = pd.read_csv("data/sample/job_descriptions.csv")


# ============================================================
# 2. Load trained model and TF-IDF vectorizer
# ============================================================

model = joblib.load(
    "src/models/resume_rank_model.pkl"
)

vectorizer = joblib.load(
    "src/models/tfidf_vectorizer.pkl"
)


# ============================================================
# 3. Select job
# ============================================================

print("\nAvailable Jobs")
print("=" * 70)

for _, available_job in jobs.iterrows():
    print(
        f"{available_job['job_id']} - "
        f"{available_job['title']}"
    )

job_id = input("\nEnter Job ID: ").strip().upper()

# Validate job ID
if job_id not in jobs["job_id"].values:
    print(f"\nError: Job ID '{job_id}' was not found.")
    raise SystemExit

job = jobs[
    jobs["job_id"] == job_id
].iloc[0]

# ============================================================
# 4. Extract required skills
# ============================================================

required_skills = extract_skills(
    job["job_description"]
)


# ============================================================
# 5. Generate candidate features
# ============================================================

results = []

for _, resume in resumes.iterrows():

    # --------------------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------------------

    resume_vector = vectorizer.transform(
        [resume["resume_text"]]
    )

    job_vector = vectorizer.transform(
        [job["job_description"]]
    )

    tfidf_similarity = cosine_similarity(
        job_vector,
        resume_vector
    )[0][0]

    # --------------------------------------------------------
    # Skill matching
    # --------------------------------------------------------

    candidate_skills = extract_skills(
        resume["resume_text"]
    )

    matched_skills = sorted(
        set(required_skills)
        & set(candidate_skills)
    )

    missing_skills = sorted(
        set(required_skills)
        - set(candidate_skills)
    )

    if required_skills:
        skill_match_ratio = (
            len(matched_skills)
            / len(required_skills)
        )
    else:
        skill_match_ratio = 0

    weighted_skill_match_ratio = (
        calculate_weighted_skill_match(
            required_skills,
            candidate_skills
        )
    )

    # --------------------------------------------------------
    # Create ML feature input
    # --------------------------------------------------------

    candidate_features = pd.DataFrame([{
        "tfidf_similarity": tfidf_similarity,
        "skill_match_ratio": skill_match_ratio,
        "weighted_skill_match_ratio": weighted_skill_match_ratio,
    }])

    # --------------------------------------------------------
    # Predict match probability
    # --------------------------------------------------------

    match_probability = model.predict_proba(
        candidate_features
    )[0][1]

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "candidate_id": resume["candidate_id"],
        "name": resume["name"],
        "match_probability": match_probability,
        "tfidf_similarity": tfidf_similarity,
        "skill_match_ratio": skill_match_ratio,
        "weighted_skill_match_ratio": weighted_skill_match_ratio,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    })


# ============================================================
# 6. Rank candidates
# ============================================================

results = sorted(
    results,
    key=lambda x: x["match_probability"],
    reverse=True
)


# ============================================================
# 7. Display ranking
# ============================================================

print("\nResume Ranking")
print("=" * 70)

print(f"\nJob: {job['title']}")

print(
    "Required Skills:",
    ", ".join(required_skills)
)

for rank, result in enumerate(
    results,
    start=1
):

    print(
        f"\n{rank}. "
        f"{result['name']} "
        f"({result['candidate_id']})"
    )

    print(
        "   Match Probability       : "
        f"{result['match_probability']:.1%}"
    )

    print(
        "   TF-IDF Similarity       : "
        f"{result['tfidf_similarity']:.3f}"
    )

    print(
        "   Skill Match Ratio       : "
        f"{result['skill_match_ratio']:.1%}"
    )

    print(
        "   Weighted Skill Match    : "
        f"{result['weighted_skill_match_ratio']:.1%}"
    )

    print(
        "   Matched Skills          :",
        ", ".join(result["matched_skills"])
        or "None"
    )

    print(
        "   Missing Skills          :",
        ", ".join(result["missing_skills"])
        or "None"
    )