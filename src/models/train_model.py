import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
training = pd.read_csv("data/sample/training_data.csv")


# ============================================================
# 2. Feature creation
# ============================================================

def create_features(training_rows, vectorizer):
    """
    Create ML features for job-candidate pairs.

    Features:
    - TF-IDF similarity
    - Skill match ratio
    - Weighted skill match ratio
    """

    features = []

    for _, row in training_rows.iterrows():

        resume = resumes[
            resumes["candidate_id"] == row["candidate_id"]
        ].iloc[0]

        job = jobs[
            jobs["job_id"] == row["job_id"]
        ].iloc[0]

        # ----------------------------------------------------
        # TF-IDF similarity
        # ----------------------------------------------------

        resume_vector = vectorizer.transform(
            [resume["resume_text"]]
        )

        job_vector = vectorizer.transform(
            [job["job_description"]]
        )

        similarity = cosine_similarity(
            job_vector,
            resume_vector
        )[0][0]

        # ----------------------------------------------------
        # Skill matching
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Store features
        # ----------------------------------------------------

        features.append({
            "tfidf_similarity": similarity,
            "skill_match_ratio": skill_match_ratio,
            "weighted_skill_match_ratio": weighted_skill_match_ratio,
            "label": row["label"],
        })

    return pd.DataFrame(features)


# ============================================================
# 3. Cross-validation setup
# ============================================================

X_rows = training.drop(columns=["label"])
y = training["label"]
groups = training["candidate_id"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

feature_columns = [
    "tfidf_similarity",
    "skill_match_ratio",
    "weighted_skill_match_ratio",
]


# ============================================================
# 4. Cross-validation
# ============================================================

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []

all_true = []
all_pred = []


for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_rows, y, groups),
    start=1
):

    train_rows = training.iloc[train_idx]
    val_rows = training.iloc[val_idx]

    # --------------------------------------------------------
    # Fit TF-IDF using training-fold resumes + job texts
    # --------------------------------------------------------

    train_candidate_ids = train_rows["candidate_id"].unique()

    train_resumes = resumes[
        resumes["candidate_id"].isin(train_candidate_ids)
    ]["resume_text"].tolist()

    job_texts = jobs["job_description"].tolist()

    tfidf_corpus = train_resumes + job_texts

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectorizer.fit(tfidf_corpus)

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    train_features = create_features(
        train_rows,
        vectorizer
    )

    val_features = create_features(
        val_rows,
        vectorizer
    )

    X_train = train_features[feature_columns]
    y_train = train_features["label"]

    X_val = val_features[feature_columns]
    y_val = val_features["label"]

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000
        )),
    ])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_val)

    y_prob = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_val,
        y_prob
    )

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    auc_scores.append(auc)

    all_true.extend(y_val)
    all_pred.extend(y_pred)

    # --------------------------------------------------------
    # Print fold results
    # --------------------------------------------------------

    print(f"\nFold {fold}")
    print("-" * 40)
    print(f"Training samples   : {len(train_rows)}")
    print(f"Validation samples : {len(val_rows)}")
    print(f"Accuracy            : {accuracy:.3f}")
    print(f"Precision           : {precision:.3f}")
    print(f"Recall              : {recall:.3f}")
    print(f"F1 Score            : {f1:.3f}")
    print(f"ROC-AUC             : {auc:.3f}")


# ============================================================
# 5. Overall cross-validation results
# ============================================================

print("\n")
print("=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

print(
    f"Mean Accuracy : {sum(accuracy_scores) / len(accuracy_scores):.3f}"
)

print(
    f"Mean Precision: {sum(precision_scores) / len(precision_scores):.3f}"
)

print(
    f"Mean Recall   : {sum(recall_scores) / len(recall_scores):.3f}"
)

print(
    f"Mean F1 Score : {sum(f1_scores) / len(f1_scores):.3f}"
)

print(
    f"Mean ROC-AUC  : {sum(auc_scores) / len(auc_scores):.3f}"
)


# ============================================================
# 6. Overall confusion matrix
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred
)

print("\nConfusion Matrix")
print("=" * 60)
print(cm)


# ============================================================
# 7. Train final model on all data
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL")
print("=" * 60)

# ------------------------------------------------------------
# Fit TF-IDF using all resumes + job descriptions
# ------------------------------------------------------------

all_resume_text = resumes["resume_text"].tolist()
all_job_text = jobs["job_description"].tolist()

final_vectorizer = TfidfVectorizer(
    stop_words="english"
)

final_vectorizer.fit(
    all_resume_text + all_job_text
)

# ------------------------------------------------------------
# Create final features
# ------------------------------------------------------------

final_features = create_features(
    training,
    final_vectorizer
)

X_final = final_features[feature_columns]
y_final = final_features["label"]

# ------------------------------------------------------------
# Final model
# ------------------------------------------------------------

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=1000
    )),
])

final_model.fit(
    X_final,
    y_final
)


# ============================================================
# 8. Final model coefficients
# ============================================================

classifier = final_model.named_steps["classifier"]

print("\nFinal Model Coefficients:")

for feature, coefficient in zip(
    feature_columns,
    classifier.coef_[0]
):
    print(
        f"{feature}: {coefficient:.3f}"
    )


# ============================================================
# 9. Save model and vectorizer
# ============================================================

joblib.dump(
    final_model,
    "src/models/resume_rank_model.pkl"
)

joblib.dump(
    final_vectorizer,
    "src/models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully.")
print("Saved: src/models/resume_rank_model.pkl")
print("Saved: src/models/tfidf_vectorizer.pkl")