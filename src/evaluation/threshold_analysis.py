import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.skills import (
    extract_skills,
    calculate_weighted_skill_match
)


# ============================================================
# 1. Load data
# ============================================================

resumes = pd.read_csv(
    "data/sample/resumes.csv"
)

jobs = pd.read_csv(
    "data/sample/job_descriptions.csv"
)

training = pd.read_csv(
    "data/sample/training_data.csv"
)


# ============================================================
# 2. Load final trained model and vectorizer
# ============================================================

model = joblib.load(
    "src/models/resume_rank_model.pkl"
)

vectorizer = joblib.load(
    "src/models/tfidf_vectorizer.pkl"
)


# ============================================================
# 3. Generate probabilities for all examples
# ============================================================

true_labels = []
probabilities = []


for _, row in training.iterrows():

    resume = resumes[
        resumes["candidate_id"]
        == row["candidate_id"]
    ].iloc[0]

    job = jobs[
        jobs["job_id"]
        == row["job_id"]
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
    # Model features
    # --------------------------------------------------------

    candidate_features = pd.DataFrame([{

        "tfidf_similarity":
            tfidf_similarity,

        "skill_match_ratio":
            skill_match_ratio,

        "weighted_skill_match_ratio":
            weighted_skill_match_ratio
    }])


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        candidate_features
    )[0][1]


    true_labels.append(
        row["label"]
    )

    probabilities.append(
        probability
    )


# ============================================================
# 4. Test different thresholds
# ============================================================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


results = []


print("\n")
print("=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 80)


for threshold in thresholds:

    predictions = [
        1 if probability >= threshold
        else 0
        for probability in probabilities
    ]


    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )


    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.3f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
    )


# ============================================================
# 5. Find best threshold by F1
# ============================================================

results_df = pd.DataFrame(
    results
)

best_result = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n")
print("=" * 80)
print("BEST THRESHOLD BY F1")
print("=" * 80)

print(
    f"Threshold : "
    f"{best_result['threshold']:.2f}"
)

print(
    f"Accuracy  : "
    f"{best_result['accuracy']:.3f}"
)

print(
    f"Precision : "
    f"{best_result['precision']:.3f}"
)

print(
    f"Recall    : "
    f"{best_result['recall']:.3f}"
)

print(
    f"F1 Score  : "
    f"{best_result['f1']:.3f}"
)