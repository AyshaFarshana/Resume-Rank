import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.preprocessing.skills import extract_skills


# ============================================================
# 1. Load data
# ============================================================

resumes = pd.read_csv("data/sample/resumes.csv")
jobs = pd.read_csv("data/sample/job_descriptions.csv")
training = pd.read_csv("data/sample/training_data.csv")


# ============================================================
# 2. Create TF-IDF vectorizer
# ============================================================

all_text = (
    resumes["resume_text"].tolist()
    + jobs["job_description"].tolist()
)

vectorizer = TfidfVectorizer(
    stop_words="english"
)

vectorizer.fit(all_text)


# ============================================================
# 3. Feature generation function
# ============================================================

def create_features(training_subset):

    features = []

    for _, row in training_subset.iterrows():

        resume = resumes[
            resumes["candidate_id"]
            == row["candidate_id"]
        ].iloc[0]

        job = jobs[
            jobs["job_id"]
            == row["job_id"]
        ].iloc[0]


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

        if required_skills:

            skill_match_ratio = (
                len(matched_skills)
                / len(required_skills)
            )

        else:

            skill_match_ratio = 0


        features.append({
            "tfidf_similarity": tfidf_similarity,
            "skill_match_ratio": skill_match_ratio,
            "label": row["label"]
        })

    return pd.DataFrame(features)


# ============================================================
# 4. Evaluation settings
# ============================================================

K = 3

precision_scores = []
recall_scores = []
average_precision_scores = []


# ============================================================
# 5. Leave-One-Job-Out evaluation
# ============================================================

for _, test_job in jobs.iterrows():

    test_job_id = test_job["job_id"]

    print("\n")
    print("=" * 65)
    print(
        f"Testing on: "
        f"{test_job['title']} ({test_job_id})"
    )
    print("=" * 65)


    # --------------------------------------------------------
    # Split by job
    # --------------------------------------------------------

    train_data = training[
        training["job_id"] != test_job_id
    ]

    test_data = training[
        training["job_id"] == test_job_id
    ]


    # --------------------------------------------------------
    # Create training features
    # --------------------------------------------------------

    train_features = create_features(
        train_data
    )

    X_train = train_features.drop(
        columns=["label"]
    )

    y_train = train_features["label"]


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Score all candidates for held-out job
    # --------------------------------------------------------

    required_skills = extract_skills(
        test_job["job_description"]
    )

    ranking_results = []

    for _, resume in resumes.iterrows():

        # TF-IDF similarity

        resume_vector = vectorizer.transform(
            [resume["resume_text"]]
        )

        job_vector = vectorizer.transform(
            [test_job["job_description"]]
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


        # ML features

        candidate_features = pd.DataFrame([{
            "tfidf_similarity": tfidf_similarity,
            "skill_match_ratio": skill_match_ratio
        }])


        # Probability

        probability = model.predict_proba(
            candidate_features
        )[0][1]


        # Actual label

        label = test_data[
            test_data["candidate_id"]
            == resume["candidate_id"]
        ]["label"].iloc[0]


        ranking_results.append({
            "candidate_id": resume["candidate_id"],
            "probability": probability,
            "label": label
        })


    # --------------------------------------------------------
    # Sort ranking
    # --------------------------------------------------------

    ranking_results = sorted(
        ranking_results,
        key=lambda x: x["probability"],
        reverse=True
    )


    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    ranked_labels = [
        result["label"]
        for result in ranking_results
    ]

    probabilities = [
        result["probability"]
        for result in ranking_results
    ]


    # Precision@K

    top_k_labels = ranked_labels[:K]

    precision_at_k = (
        sum(top_k_labels)
        / K
    )


    # Recall@K

    total_relevant = sum(
        ranked_labels
    )

    recall_at_k = (
        sum(top_k_labels)
        / total_relevant
        if total_relevant > 0
        else 0
    )


    # Average Precision

    average_precision = average_precision_score(
        ranked_labels,
        probabilities
    )


    # Store scores

    precision_scores.append(
        precision_at_k
    )

    recall_scores.append(
        recall_at_k
    )

    average_precision_scores.append(
        average_precision
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        f"\nPrecision@{K}: "
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

    print("\nRanking:")

    for rank, result in enumerate(
        ranking_results,
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
# 6. Overall results
# ============================================================

print("\n")
print("=" * 65)
print("LEAVE-ONE-JOB-OUT RESULTS")
print("=" * 65)

print(
    f"Mean Precision@{K}: "
    f"{sum(precision_scores) / len(precision_scores):.3f}"
)

print(
    f"Mean Recall@{K}: "
    f"{sum(recall_scores) / len(recall_scores):.3f}"
)

print(
    "Mean Average Precision (MAP): "
    f"{sum(average_precision_scores) / len(average_precision_scores):.3f}"
)