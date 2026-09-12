# Resume Rank
Link: https://resume-rank-ycqmemal67nalndzduqict.streamlit.app/

Resume Rank is a machine learning application that ranks candidate resumes against a job description and provides an interpretable match score.

The system combines text similarity with explicit skill matching to estimate how well a candidate matches a particular role.

## Features

- Resume ranking based on job requirements
- TF-IDF text similarity
- Skill matching
- Weighted skill matching
- Logistic Regression classification model
- Match probability for each candidate
- Matched and missing skill identification
- PDF and TXT resume upload
- Sample resume dataset
- Streamlit web interface
- Cross-validation and out-of-fold evaluation
- Error analysis and threshold analysis

## How It Works

The system follows this pipeline:

```text
Job Description
       │
       ▼
Required Skill Extraction
       │
       ├───────────────┐
       │               │
       ▼               ▼
Candidate Resume   Job Description
       │               │
       └───────┬───────┘
               ▼
        TF-IDF Similarity
               │
               ├── Skill Match Ratio
               │
               └── Weighted Skill Match Ratio
                       │
                       ▼
              Logistic Regression
                       │
                       ▼
              Match Probability
                       │
                       ▼
                Resume Ranking
