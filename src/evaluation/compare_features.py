import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.preprocessing.skills import (
    extract_skills,
    calculate_weighted_skill_match
)


# ============================================================
# Load data
# ============================================================

resumes = pd.read_csv("data/sample/resumes.csv")
jobs = pd.read_csv("data/sample/job_descriptions.csv")
training = pd.read_csv("data/sample/training_data.csv")


# ============================================================
# Feature creation
# ============================================================

def create_features(training_rows, vectorizer):

    features = []

    for _, row in training_rows.iterrows():

        resume = resumes[
            resumes["candidate_id"] == row["candidate_id"]
        ].iloc[0]

        job = jobs[
            jobs["job_id"] == row["job_id"]
        ].iloc[0]


        # TF-IDF similarity

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


        # Skills

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


        # Normal skill ratio

        if required_skills:

            skill_match_ratio = (
                len(matched_skills)
                / len(required_skills)
            )

        else:

            skill_match_ratio = 0


        # Weighted skill ratio

        weighted_skill_match_ratio = (
            calculate_weighted_skill_match(
                required_skills,
                candidate_skills
            )
        )


        features.append({

            "tfidf_similarity": similarity,

            "skill_match_ratio": skill_match_ratio,

            "weighted_skill_match_ratio":
                weighted_skill_match_ratio,

            "label": row["label"]
        })


    return pd.DataFrame(features)


# ============================================================
# Feature configurations
# ============================================================

feature_sets = {

    "TF-IDF + Skill Ratio": [
        "tfidf_similarity",
        "skill_match_ratio"
    ],

    "TF-IDF + Weighted Skill": [
        "tfidf_similarity",
        "weighted_skill_match_ratio"
    ],

    "TF-IDF + Both Skill Features": [
        "tfidf_similarity",
        "skill_match_ratio",
        "weighted_skill_match_ratio"
    ]
}


# ============================================================
# Cross-validation setup
# ============================================================

X_rows = training.drop(columns=["label"])

y = training["label"]

groups = training["candidate_id"]


cv = StratifiedGroupKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


# ============================================================
# Compare feature sets
# ============================================================

comparison_results = []


for feature_set_name, feature_columns in feature_sets.items():

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []


    print("\n")
    print("=" * 70)

    print(feature_set_name)

    print("=" * 70)


    for fold, (train_idx, val_idx) in enumerate(

        cv.split(X_rows, y, groups),

        start=1

    ):

        train_rows = training.iloc[train_idx]

        val_rows = training.iloc[val_idx]


        # Fit TF-IDF on training fold

        train_candidate_ids = (
            train_rows["candidate_id"]
            .unique()
        )

        train_resumes = resumes[

            resumes["candidate_id"]
            .isin(train_candidate_ids)

        ]["resume_text"].tolist()


        all_job_text = (
            jobs["job_description"]
            .tolist()
        )


        vectorizer = TfidfVectorizer(

            stop_words="english"

        )


        vectorizer.fit(

            train_resumes
            + all_job_text

        )


        # Create features

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


        # Model

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


        # Predictions

        y_pred = model.predict(X_val)

        y_prob = model.predict_proba(

            X_val

        )[:, 1]


        # Metrics

        accuracy_scores.append(

            accuracy_score(

                y_val,

                y_pred

            )

        )


        precision_scores.append(

            precision_score(

                y_val,

                y_pred,

                zero_division=0

            )

        )


        recall_scores.append(

            recall_score(

                y_val,

                y_pred,

                zero_division=0

            )

        )


        f1_scores.append(

            f1_score(

                y_val,

                y_pred,

                zero_division=0

            )

        )


        auc_scores.append(

            roc_auc_score(

                y_val,

                y_prob

            )

        )


        print(

            f"Fold {fold}: "

            f"Accuracy={accuracy_scores[-1]:.3f}, "

            f"F1={f1_scores[-1]:.3f}, "

            f"ROC-AUC={auc_scores[-1]:.3f}"

        )


    # Store mean results

    comparison_results.append({

        "Feature Set": feature_set_name,

        "Accuracy":
            sum(accuracy_scores)
            / len(accuracy_scores),

        "Precision":
            sum(precision_scores)
            / len(precision_scores),

        "Recall":
            sum(recall_scores)
            / len(recall_scores),

        "F1":
            sum(f1_scores)
            / len(f1_scores),

        "ROC-AUC":
            sum(auc_scores)
            / len(auc_scores)

    })


# ============================================================
# Final comparison table
# ============================================================

results_df = pd.DataFrame(

    comparison_results

)


print("\n")
print("=" * 70)

print("FEATURE COMPARISON RESULTS")

print("=" * 70)

print(

    results_df.to_string(

        index=False

    )

)