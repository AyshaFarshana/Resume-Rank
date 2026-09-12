import pandas as pd

from src.preprocessing.skills import extract_skills


# ============================================================
# 1. Load data
# ============================================================

resumes = pd.read_csv("data/sample/resumes.csv")
jobs = pd.read_csv("data/sample/job_descriptions.csv")
training = pd.read_csv("data/sample/training_data.csv")


# ============================================================
# 2. Audit thresholds
# ============================================================

# Positive candidate with <= 20% skill match
LOW_MATCH_POSITIVE_THRESHOLD = 0.20

# Negative candidate with >= 40% skill match
HIGH_MATCH_NEGATIVE_THRESHOLD = 0.40


suspicious_positives = []
suspicious_negatives = []


# ============================================================
# 3. Check every training example
# ============================================================

for _, row in training.iterrows():

    job = jobs[
        jobs["job_id"] == row["job_id"]
    ].iloc[0]

    resume = resumes[
        resumes["candidate_id"] == row["candidate_id"]
    ].iloc[0]


    # Extract skills

    required_skills = extract_skills(
        job["job_description"]
    )

    candidate_skills = extract_skills(
        resume["resume_text"]
    )


    # Calculate matched and missing skills

    matched_skills = sorted(
        set(required_skills)
        & set(candidate_skills)
    )

    missing_skills = sorted(
        set(required_skills)
        - set(candidate_skills)
    )


    # Skill match ratio

    if required_skills:

        skill_match_ratio = (
            len(matched_skills)
            / len(required_skills)
        )

    else:

        skill_match_ratio = 0


    # ========================================================
    # Flag suspicious examples
    # ========================================================

    # Positive label but poor skill match
    if (
        row["label"] == 1
        and skill_match_ratio
        <= LOW_MATCH_POSITIVE_THRESHOLD
    ):

        suspicious_positives.append({
            "job_id": row["job_id"],
            "job_title": job["title"],
            "candidate_id": row["candidate_id"],
            "candidate_name": resume["name"],
            "label": row["label"],
            "skill_match_ratio": skill_match_ratio,
            "matched_skills": ", ".join(matched_skills) or "None",
            "missing_skills": ", ".join(missing_skills) or "None"
        })


    # Negative label but strong skill match
    if (
        row["label"] == 0
        and skill_match_ratio
        >= HIGH_MATCH_NEGATIVE_THRESHOLD
    ):

        suspicious_negatives.append({
            "job_id": row["job_id"],
            "job_title": job["title"],
            "candidate_id": row["candidate_id"],
            "candidate_name": resume["name"],
            "label": row["label"],
            "skill_match_ratio": skill_match_ratio,
            "matched_skills": ", ".join(matched_skills) or "None",
            "missing_skills": ", ".join(missing_skills) or "None"
        })


# ============================================================
# 4. Display suspicious positives
# ============================================================

print("\n")
print("=" * 80)
print("SUSPICIOUS POSITIVE EXAMPLES")
print("=" * 80)

if suspicious_positives:

    for example in suspicious_positives:

        print(
            f"\nJob       : "
            f"{example['job_title']} "
            f"({example['job_id']})"
        )

        print(
            f"Candidate : "
            f"{example['candidate_name']} "
            f"({example['candidate_id']})"
        )

        print(
            f"Label     : "
            f"{example['label']}"
        )

        print(
            f"Skill Match Ratio : "
            f"{example['skill_match_ratio']:.1%}"
        )

        print(
            f"Matched Skills    : "
            f"{example['matched_skills']}"
        )

        print(
            f"Missing Skills    : "
            f"{example['missing_skills']}"
        )

else:

    print("\nNo suspicious positive examples found.")


# ============================================================
# 5. Display suspicious negatives
# ============================================================

print("\n")
print("=" * 80)
print("SUSPICIOUS NEGATIVE EXAMPLES")
print("=" * 80)

if suspicious_negatives:

    for example in suspicious_negatives:

        print(
            f"\nJob       : "
            f"{example['job_title']} "
            f"({example['job_id']})"
        )

        print(
            f"Candidate : "
            f"{example['candidate_name']} "
            f"({example['candidate_id']})"
        )

        print(
            f"Label     : "
            f"{example['label']}"
        )

        print(
            f"Skill Match Ratio : "
            f"{example['skill_match_ratio']:.1%}"
        )

        print(
            f"Matched Skills    : "
            f"{example['matched_skills']}"
        )

        print(
            f"Missing Skills    : "
            f"{example['missing_skills']}"
        )

else:

    print("\nNo suspicious negative examples found.")


# ============================================================
# 6. Summary
# ============================================================

print("\n")
print("=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)

print(
    "Suspicious positives:",
    len(suspicious_positives)
)

print(
    "Suspicious negatives:",
    len(suspicious_negatives)
)