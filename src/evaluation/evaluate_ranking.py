import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score

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
# 3. Ranking evaluation settings
# ============================================================

K = 3

precision_at_k_scores = []
recall_at_k_scores = []
average_precision_scores = []


# ============================================================
# 4. Evaluate each job
# ============================================================

for _, job in jobs.iterrows():

    job_id = job["job_id"]

    required_skills = extract_skills(
        job["job_description"]
    )

    job_results = []


    # --------------------------------------------------------
    # Score every candidate for this job
    # --------------------------------------------------------

    for _, resume in resumes.iterrows():

        # TF-IDF similarity

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


        # Skill matching

        candidate_skills = extract_skills(
            resume["resume_text"]
        )

        matched_skills = (
            set(required_skills)
            & set(candidate_skills)
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

        # ML feature input

        candidate_features = pd.DataFrame([{
            "tfidf_similarity": tfidf_similarity,
            "skill_match_ratio": skill_match_ratio,
            "weighted_skill_match_ratio":
                weighted_skill_match_ratio
        }])


        # Match probability

        match_probability = model.predict_proba(
            candidate_features
        )[0][1]


        # Get actual label

        label_row = training[
            (training["job_id"] == job_id)
            &
            (
                training["candidate_id"]
                == resume["candidate_id"]
            )
        ]

        actual_label = label_row[
            "label"
        ].iloc[0]


        job_results.append({
            "candidate_id": resume["candidate_id"],
            "probability": match_probability,
            "label": actual_label
        })


    # --------------------------------------------------------
    # Sort candidates by probability
    # --------------------------------------------------------

    job_results = sorted(
        job_results,
        key=lambda x: x["probability"],
        reverse=True
    )


    # --------------------------------------------------------
    # Extract ranking labels
    # --------------------------------------------------------

    ranked_labels = [
        result["label"]
        for result in job_results
    ]


    # --------------------------------------------------------
    # Precision@K
    # --------------------------------------------------------

    top_k_labels = ranked_labels[:K]

    precision_at_k = (
        sum(top_k_labels)
        / K
    )


    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    total_relevant = sum(ranked_labels)

    if total_relevant > 0:

        recall_at_k = (
            sum(top_k_labels)
            / total_relevant
        )

    else:

        recall_at_k = 0


    # --------------------------------------------------------
    # Average Precision
    # --------------------------------------------------------

    probabilities = [
        result["probability"]
        for result in job_results
    ]

    average_precision = average_precision_score(
        ranked_labels,
        probabilities
    )


    # Store metrics

    precision_at_k_scores.append(
        precision_at_k
    )

    recall_at_k_scores.append(
        recall_at_k
    )

    average_precision_scores.append(
        average_precision
    )


    # --------------------------------------------------------
    # Print job results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(
        f"{job['title']} ({job_id})"
    )
    print("=" * 60)

    print(
        f"Precision@{K}: "
        f"{precision_at_k:.3f}"
    )

    print(
        f"Recall@{K}: "
        f"{recall_at_k:.3f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.3f}"
    )

    print("\nTop Candidates:")

    for rank, result in enumerate(
        job_results[:K],
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['candidate_id']} "
            f"| Probability: "
            f"{result['probability']:.1%} "
            f"| Label: "
            f"{result['label']}"
        )


# ============================================================
# 5. Overall ranking metrics
# ============================================================

mean_precision_at_k = (
    sum(precision_at_k_scores)
    / len(precision_at_k_scores)
)

mean_recall_at_k = (
    sum(recall_at_k_scores)
    / len(recall_at_k_scores)
)

mean_average_precision = (
    sum(average_precision_scores)
    / len(average_precision_scores)
)


print("\n")
print("=" * 60)
print("OVERALL RANKING EVALUATION")
print("=" * 60)

print(
    f"Mean Precision@{K}: "
    f"{mean_precision_at_k:.3f}"
)

print(
    f"Mean Recall@{K}: "
    f"{mean_recall_at_k:.3f}"
)

print(
    f"Mean Average Precision (MAP): "
    f"{mean_average_precision:.3f}"
)