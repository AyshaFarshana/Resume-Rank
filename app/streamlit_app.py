import os
import sys

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Allow imports from src/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_PATH)

from preprocessing.resume_parser import extract_resume_text
from preprocessing.skills import (
    extract_skills,
    calculate_weighted_skill_match,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Resume Rank",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>

        /* =========================================
           MAIN PAGE
           ========================================= */

        .main {
            padding-top: 1rem;
        }


        /* =========================================
           HEADER
           ========================================= */

        .title {
            font-size: 2.7rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }

        .subtitle {
            font-size: 1.05rem;
            opacity: 0.65;
            margin-bottom: 2rem;
        }


        /* =========================================
           SIDEBAR
           ========================================= */

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.30);
        }


        /* =========================================
           SKILL TAGS
           ========================================= */

        .skill {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            margin: 0.2rem 0.25rem 0.2rem 0;
            border-radius: 6px;
            border: 1px solid rgba(128, 128, 128, 0.35);
            font-size: 0.85rem;
        }


        /* =========================================
           UPLOAD AREA
           ========================================= */

        [data-testid="stFileUploader"] {
            padding: 0.5rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 10px;
        }


        /* =========================================
           BUTTON
           ========================================= */

        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }


        /* =========================================
        RANKING CARDS
        ========================================= */

        .rank-card {
            padding: 1.2rem 1.3rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            margin-bottom: 1rem;
        }

        .rank-number {
            font-size: 1.4rem;
            font-weight: 700;
            opacity: 0.65;
        }

        .candidate-name {
            font-size: 1.2rem;
            font-weight: 650;
        }

        .match-status {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            opacity: 0.75;
        }

        .score-large {
            font-size: 1.8rem;
            font-weight: 700;
            text-align: right;
        }

        .score-label {
            font-size: 0.75rem;
            opacity: 0.6;
            text-align: right;
        }

        .metric-label {
            font-size: 0.78rem;
            opacity: 0.65;
        }

        .skill-section {
            margin-top: 0.7rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load data and model
# --------------------------------------------------

@st.cache_data
def load_data():
    resumes = pd.read_csv("data/sample/resumes.csv")
    jobs = pd.read_csv("data/sample/job_descriptions.csv")
    return resumes, jobs


@st.cache_resource
def load_model():
    model = joblib.load("src/models/resume_rank_model.pkl")
    vectorizer = joblib.load("src/models/tfidf_vectorizer.pkl")
    return model, vectorizer


resumes_df, jobs_df = load_data()
model, vectorizer = load_model()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "ranking_results" not in st.session_state:
    st.session_state.ranking_results = None

if "previous_job_id" not in st.session_state:
    st.session_state.previous_job_id = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title"> Resume Rank</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered resume screening and job matching"
    "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Job Settings")

    job_id = st.selectbox(
        "Select a job",
        jobs_df["job_id"].tolist(),
        format_func=lambda x: (
            f"{x} — "
            f"{jobs_df.loc[jobs_df['job_id'] == x, 'title'].iloc[0]}"
        ),
    )

    st.divider()

    st.markdown("### Model")
    st.write("Logistic Regression")

    st.markdown("### Features")
    st.write("• TF-IDF similarity")
    st.write("• Skill match")
    st.write("• Weighted skill match")

    st.divider()

    st.caption("Decision threshold: 0.50")


# --------------------------------------------------
# Clear previous results when job changes
# --------------------------------------------------

if st.session_state.previous_job_id != job_id:
    st.session_state.ranking_results = None
    st.session_state.previous_job_id = job_id


# --------------------------------------------------
# Job information
# --------------------------------------------------

job = jobs_df[jobs_df["job_id"] == job_id].iloc[0]

required_skills = extract_skills(job["job_description"])

col1, col2 = st.columns([1.8, 1])

with col1:
    st.markdown("###  Job Description")
    st.markdown(f"**{job['title']}**")
    st.write(job["job_description"])

with col2:
    st.markdown("### Required Skills")

    skills_html = "".join(
    f'<span class="skill">{skill}</span>'
    for skill in required_skills
    )

    st.markdown(
    skills_html,
    unsafe_allow_html=True,
    )


st.divider()


# --------------------------------------------------
# Resume upload
# --------------------------------------------------

st.markdown("###  Resume Input")

with st.container(border=True):
    uploaded_files = st.file_uploader(
    "Upload resumes",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload one or more PDF or TXT resumes.",
    )

use_sample_data = use_sample_data = st.checkbox(
    "Use sample resumes",
    value=not uploaded_files,
    help="Use the project's built-in sample resumes for testing.",
)


# --------------------------------------------------
# Rank button
# --------------------------------------------------

rank_clicked = st.button(
    " Rank Resumes",
    type="primary",
    key="rank_resumes_button",
)


if rank_clicked:

    candidate_records = []

    # ----------------------------------------------
    # Uploaded resumes
    # ----------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            try:
                resume_text = extract_resume_text(uploaded_file)

                candidate_records.append(
                    {
                        "candidate_id": uploaded_file.name,
                        "candidate_name": uploaded_file.name,
                        "resume_text": resume_text,
                    }
                )

            except Exception as e:

                st.error(
                    f"Could not process {uploaded_file.name}: {e}"
                )

    # ----------------------------------------------
    # Sample resumes
    # ----------------------------------------------

    elif use_sample_data:

        for _, row in resumes_df.iterrows():

            candidate_records.append(
                {
                    "candidate_id": row["candidate_id"],
                    "candidate_name": row["name"],
                    "resume_text": row["resume_text"],
                }
            )

    else:

        st.warning(
            "Please upload a resume or select sample resumes."
        )

    # ----------------------------------------------
    # Ranking
    # ----------------------------------------------

    results = []

    job_vector = vectorizer.transform(
        [job["job_description"]]
    )

    for candidate in candidate_records:

        resume_text = candidate["resume_text"]

        resume_vector = vectorizer.transform(
            [resume_text]
        )

        tfidf_similarity = cosine_similarity(
            resume_vector,
            job_vector,
        )[0][0]

        candidate_skills = extract_skills(resume_text)

        if required_skills:
            skill_match = len(
                set(required_skills) & set(candidate_skills)
            ) / len(set(required_skills))
        else:
            skill_match = 0

        weighted_skill_match = calculate_weighted_skill_match(
            required_skills,
            candidate_skills,
)

        features = pd.DataFrame(
            [
                {
                    "tfidf_similarity": tfidf_similarity,
                    "skill_match_ratio": skill_match,
                    "weighted_skill_match_ratio": weighted_skill_match,
                }
            ]
        )

        probability = model.predict_proba(
            features
        )[0][1]

        matched_skills = sorted(
            set(required_skills) & set(candidate_skills)
        )

        missing_skills = sorted(
            set(required_skills) - set(candidate_skills)
        )

        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_name": candidate["candidate_name"],
                "probability": probability,
                "tfidf_similarity": tfidf_similarity,
                "skill_match": skill_match,
                "weighted_skill_match": weighted_skill_match,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
            }
        )

    results = sorted(
        results,
        key=lambda x: x["probability"],
        reverse=True,
    )

    st.session_state.ranking_results = results


# --------------------------------------------------
# Display results
# --------------------------------------------------

if st.session_state.ranking_results:

    results = st.session_state.ranking_results

    st.divider()

    st.markdown("## Ranking Results")

    total_candidates = len(results)

    matched_candidates = sum(
        r["probability"] >= 0.50
        for r in results
    )

    average_score = sum(
        r["probability"]
        for r in results
    ) / total_candidates

    st.caption(
        f"Ranked {total_candidates} candidate(s) for {job['title']} "
        "using the trained ML model."
    )

    # ----------------------------------------------
    # Summary metrics
    # ----------------------------------------------

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Candidates",
            total_candidates,
        )

    with m2:
        st.metric(
            "Above Threshold",
            matched_candidates,
        )

    with m3:
        st.metric(
            "Average Match",
            f"{average_score:.1%}",
        )

    st.divider()



    # ----------------------------------------------
    # Candidate ranking cards
    # ----------------------------------------------

    for rank, result in enumerate(results, start=1):

        probability = result["probability"]

        if probability >= 0.30:
            status = "MATCH"
        else:
            status = "BELOW THRESHOLD"

        with st.container(border=True):

            # ------------------------------------------
            # Candidate header
            # ------------------------------------------

            col_rank, col_name, col_score = st.columns(
                [0.7, 3.0, 1.2]
            )

            with col_rank:
                st.markdown(
                    f'<div class="rank-number">#{rank}</div>',
                    unsafe_allow_html=True,
                )

            with col_name:
                st.markdown(
                    f'<div class="candidate-name">'
                    f'{result["candidate_name"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="match-status">{status}</div>',
                    unsafe_allow_html=True,
                )

            with col_score:
                st.markdown(
                    f'<div class="score-large">'
                    f'{probability:.1%}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="score-label">match probability</div>',
                    unsafe_allow_html=True,
                )


            # ------------------------------------------
            # Probability bar
            # ------------------------------------------

            st.progress(
                min(max(probability, 0.0), 1.0)
            )


            # ------------------------------------------
            # Model features
            # ------------------------------------------

            st.markdown("**Match Signals**")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "TF-IDF Similarity",
                    f'{result["tfidf_similarity"]:.1%}',
                )

            with metric2:
                st.metric(
                    "Skill Match",
                    f'{result["skill_match"]:.1%}',
                )

            with metric3:
                st.metric(
                    "Weighted Skill Match",
                    f'{result["weighted_skill_match"]:.1%}',
                )


            # ------------------------------------------
            # Skills
            # ------------------------------------------

            skill_col1, skill_col2 = st.columns(2)

            with skill_col1:

                st.markdown("**Matched Skills**")

                if result["matched_skills"]:

                    skills_html = "".join(
                        f'<span class="skill">'
                        f'✓ {skill}'
                        f'</span>'
                        for skill in result["matched_skills"]
                    )

                    st.markdown(
                        skills_html,
                        unsafe_allow_html=True,
                    )

                else:
                    st.caption("No required skills detected")


            with skill_col2:

                st.markdown("**Missing Skills**")

                if result["missing_skills"]:

                    skills_html = "".join(
                        f'<span class="skill">'
                        f'✗ {skill}'
                        f'</span>'
                        for skill in result["missing_skills"]
                    )

                    st.markdown(
                        skills_html,
                        unsafe_allow_html=True,
                    )

                else:
                    st.caption("No missing required skills")

    # ----------------------------------------------
    # Ranking table
    # ----------------------------------------------

    st.markdown("### Ranking Overview")
    st.caption(
        "Detailed ranking scores for all evaluated candidates."
    )

    table_data = []

    for rank, result in enumerate(results, start=1):

        table_data.append(
            {
                "Rank": rank,
                "Candidate": result["candidate_name"],
                "Match Probability": f"{result['probability']:.1%}",
                "TF-IDF": f"{result['tfidf_similarity']:.1%}",
                "Skill Match": f"{result['skill_match']:.1%}",
                "Weighted Skill Match": (
                    f"{result['weighted_skill_match']:.1%}"
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )