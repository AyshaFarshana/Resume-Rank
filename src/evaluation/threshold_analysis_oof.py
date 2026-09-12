import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.preprocessing.skills import (
    extract_skills,
    calculate_weighted_skill_match
)


# ============================================================
# LOAD DATA
# ============================================================

training_data = pd.read_csv("data/sample/training_data.csv")
resumes = pd.read_csv("data/sample/resumes.csv")
jobs = pd.read_csv("data/sample/job_descriptions.csv")


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(rows, vectorizer):

    candidate_texts = []
    job_texts = []

    skill_ratios = []
    weighted_skill_ratios = []
    labels = []

    for _, row in rows.iterrows():

        candidate = resumes[
            resumes["candidate_id"] == row["candidate_id"]
        ].iloc[0]

        job = jobs[
            jobs["job_id"] == row["job_id"]
        ].iloc[0]

        candidate_text = candidate["resume_text"]
        job_text = job["job_description"]

        candidate_texts.append(candidate_text)
        job_texts.append(job_text)

        required_skills = extract_skills(job_text)
        candidate_skills = extract_skills(candidate_text)

        matched_skills = set(required_skills) & set(candidate_skills)

        skill_ratio = (
            len(matched_skills) / len(set(required_skills))
            if required_skills
            else 0
        )

        weighted_skill_ratio = calculate_weighted_skill_match(
            required_skills,
            candidate_skills
        )

        skill_ratios.append(skill_ratio)
        weighted_skill_ratios.append(weighted_skill_ratio)

        labels.append(row["label"])

    candidate_vectors = vectorizer.transform(candidate_texts)
    job_vectors = vectorizer.transform(job_texts)

    similarities = np.asarray(
        candidate_vectors.multiply(job_vectors).sum(axis=1)
    ).ravel()

    return pd.DataFrame({
        "tfidf_similarity": similarities,
        "skill_match_ratio": skill_ratios,
        "weighted_skill_match_ratio": weighted_skill_ratios,
        "label": labels
    })


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================

X_labels = training_data["label"]
groups = training_data["candidate_id"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = np.zeros(len(training_data))
oof_labels = training_data["label"].values


for fold, (train_idx, val_idx) in enumerate(
    cv.split(training_data, X_labels, groups),
    start=1
):

    train_rows = training_data.iloc[train_idx]
    val_rows = training_data.iloc[val_idx]

    # --------------------------------------------------------
    # Fit TF-IDF only on training candidates + job descriptions
    # --------------------------------------------------------

    train_candidate_texts = []

    for candidate_id in train_rows["candidate_id"].unique():

        candidate = resumes[
            resumes["candidate_id"] == candidate_id
        ].iloc[0]

        train_candidate_texts.append(candidate["resume_text"])

    all_job_texts = jobs["job_description"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectorizer.fit(
        train_candidate_texts + all_job_texts
    )

    # --------------------------------------------------------
    # Create train and validation features
    # --------------------------------------------------------

    train_features = create_features(
        train_rows,
        vectorizer
    )

    val_features = create_features(
        val_rows,
        vectorizer
    )

    feature_columns = [
        "tfidf_similarity",
        "skill_match_ratio",
        "weighted_skill_match_ratio"
    ]

    X_train = train_features[feature_columns]
    y_train = train_features["label"]

    X_val = val_features[feature_columns]

    # --------------------------------------------------------
    # Scale + train model
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Store OUT-OF-FOLD probabilities
    # --------------------------------------------------------

    oof_probabilities[val_idx] = model.predict_proba(
        X_val_scaled
    )[:, 1]

    print(f"Fold {fold} complete")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n")
print("=" * 80)
print("OUT-OF-FOLD THRESHOLD ANALYSIS")
print("=" * 80)

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

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 80)

for threshold in thresholds:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        oof_labels,
        predictions
    )

    precision = precision_score(
        oof_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        oof_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        oof_labels,
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
# BEST THRESHOLD
# ============================================================

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n")
print("=" * 80)
print("BEST OUT-OF-FOLD THRESHOLD BY F1")
print("=" * 80)

print(f"Threshold : {best_row['threshold']:.2f}")
print(f"Accuracy  : {best_row['accuracy']:.3f}")
print(f"Precision : {best_row['precision']:.3f}")
print(f"Recall    : {best_row['recall']:.3f}")
print(f"F1 Score  : {best_row['f1']:.3f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/oof_threshold_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/oof_threshold_results.csv")