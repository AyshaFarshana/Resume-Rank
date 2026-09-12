import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.skills import (
    extract_skills,
    calculate_weighted_skill_match
)


# ============================================================
# 1. Load data
# ============================================================

resumes = pd.read_csv("data/sample/resumes.csv")
jobs = pd.read_csv("data/sample/job_descriptions.csv")
training = pd.read_csv("data/sample/training_data.csv")


# ============================================================
# 2. Load trained model and vectorizer
# ============================================================

model = joblib.load(
    "src/models/resume_rank_model.pkl"
)

vectorizer = joblib.load(
    "src/models/tfidf_vectorizer.pkl"
)


# ============================================================
# 3. Analyze every job-candidate pair
# ============================================================

analysis_results = []


for _, row in training.iterrows():

    resume = resumes[
        resumes["candidate_id"] == row["candidate_id"]
    ].iloc[0]

    job = jobs[
        jobs["job_id"] == row["job_id"]
    ].iloc[0]


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
    # Skill features
    # --------------------------------------------------------

    required_skills = extract_skills(
        job["job_description"]
    )

    candidate_skills = extract_skills(
        resume["resume_text"]
    )

    matched_skills = (
        set(required_skills)
        & set(candidate_skills)
    )

    missing_skills = (
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
    # Model prediction
    # --------------------------------------------------------

    candidate_features = pd.DataFrame([{
        "tfidf_similarity": tfidf_similarity,
        "skill_match_ratio": skill_match_ratio,
        "weighted_skill_match_ratio":
            weighted_skill_match_ratio
    }])


    probability = model.predict_proba(
        candidate_features
    )[0][1]


    prediction = model.predict(
        candidate_features
    )[0]


    analysis_results.append({

        "job_id": row["job_id"],
        "job_title": job["title"],

        "candidate_id": resume["candidate_id"],
        "candidate_name": resume["name"],

        "actual_label": row["label"],
        "prediction": prediction,

        "probability": probability,

        "tfidf_similarity": tfidf_similarity,

        "skill_match_ratio": skill_match_ratio,

        "weighted_skill_match_ratio":
            weighted_skill_match_ratio,

        "matched_skills":
            ", ".join(sorted(matched_skills))
            if matched_skills else "None",

        "missing_skills":
            ", ".join(sorted(missing_skills))
            if missing_skills else "None"
    })


# ============================================================
# 4. Create results DataFrame
# ============================================================

results_df = pd.DataFrame(
    analysis_results
)


# ============================================================
# 5. False Positives
# ============================================================

false_positives = results_df[
    (results_df["actual_label"] == 0)
    &
    (results_df["prediction"] == 1)
]


print("\n")
print("=" * 80)
print("FALSE POSITIVES")
print("=" * 80)

if false_positives.empty:

    print("No false positives found.")

else:

    print(
        false_positives[
            [
                "job_title",
                "candidate_name",
                "probability",
                "tfidf_similarity",
                "skill_match_ratio",
                "weighted_skill_match_ratio",
                "matched_skills",
                "missing_skills"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 6. False Negatives
# ============================================================

false_negatives = results_df[
    (results_df["actual_label"] == 1)
    &
    (results_df["prediction"] == 0)
]


print("\n")
print("=" * 80)
print("FALSE NEGATIVES")
print("=" * 80)

if false_negatives.empty:

    print("No false negatives found.")

else:

    print(
        false_negatives[
            [
                "job_title",
                "candidate_name",
                "probability",
                "tfidf_similarity",
                "skill_match_ratio",
                "weighted_skill_match_ratio",
                "matched_skills",
                "missing_skills"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 7. Errors grouped by job
# ============================================================

errors = results_df[
    results_df["actual_label"]
    != results_df["prediction"]
]


print("\n")
print("=" * 80)
print("ERRORS BY JOB")
print("=" * 80)

if errors.empty:

    print("No classification errors found.")

else:

    print(
        errors.groupby("job_title")
        .size()
        .sort_values(ascending=False)
    )


# ============================================================
# 8. Save analysis results
# ============================================================

results_df.to_csv(
    "data/error_analysis_results.csv",
    index=False
)

print("\n")
print("Full error analysis saved to:")
print("data/error_analysis_results.csv")