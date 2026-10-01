# 🎯 AI Career Navigator

A machine learning and NLP application that extracts structured candidate profiles from uploaded PDF resumes, retrieves live jobs from the **Arbeitnow Job Board API**, computes exact feature representations, and ranks career matches using the trained machine learning pipeline from `final_project.ipynb`.

---

## 🏛️ Architecture Overview

The application is built strictly around the ML development pipeline developed in `final_project.ipynb`:

```text
                      final_project.ipynb
                              │
                    TRAINED MODEL & NLP ARTIFACTS
                              │
                              ▼
PDF Resume ──> CV Extractor ──> Candidate Profile
                                       │
                                       ▼
                              Feature Engineering (make_features)
                                       │
Live Arbeitnow API ──> Cleaner ──> Job Profile
                                       │
                                       ▼
                              Exact 19 Features
                                       │
                                       ▼
                                 Trained Model
                              ┌────────┴────────┐
                              ▼                 ▼
                         Prediction        Confidence
                              │
                              ├──────── Shared Skills (Candidate ∩ Job)
                              │
                              └──────── Missing Skills (Job − Candidate)
                                       │
                                       ▼
                                Ranked Job Cards
                                       │
                                       ▼
                                Streamlit UI (app.py)
```

---

## 📂 Project Structure

```text
d:\project_mfjm\
├── final_project.ipynb        # Source of truth for ML training & NLP pipeline
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation
├── candidate_job_pairs.csv    # Training dataset
│
├── models/                    # Saved ML & NLP artifacts
│   ├── ai_resume_job_match_model.joblib
│   └── ai_resume_job_match_nlp_artifacts.joblib
│
├── src/                       # Production modules mirroring the notebook
│   ├── __init__.py
│   ├── nlp_utils.py           # Text cleaning, stopwords, education, seniority, experience
│   ├── artifacts.py           # Loads model, TF-IDF vectorizers, vocabularies
│   ├── feature_engineering.py # exact make_features() & pair_cosine() (19 features)
│   ├── cv_extractor.py        # PDF text extraction and candidate profile validation
│   ├── job_api.py             # Arbeitnow API client with error handling & deduplication
│   ├── job_cleaner.py         # HTML/Markdown stripping, whitespace normalization
│   ├── skill_matching.py      # Case-insensitive shared/missing skills computation
│   └── matcher.py             # Inference loop, sorting, confidence calculation
│
├── data/
│   └── sample_cv.pdf          # Sample resume for testing and demonstration
│
├── test_suite.py              # Unit & validation test suite
└── test_end_to_end.py         # End-to-end integration test with live API
```

---

## ⚙️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required core packages:
- `streamlit`
- `xgboost`
- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`
- `requests`
- `beautifulsoup4`
- `pdfplumber` / `pypdf`
- `rapidfuzz`
- `reportlab`

### 2. Export / Prepare Artifacts (If needed)

The model and NLP artifacts are pre-generated from `final_project.ipynb` and stored in `models/`.
If you ever retrain the notebook, run:

```bash
python train_and_export.py
```

---

## 🚀 Running the Streamlit Application

Launch the application with:

```bash
streamlit run app.py
```

The app will start at `http://localhost:8501`.

---

## 🔍 How the System Works

### 1. Model Loading
The trained model (`XGBClassifier`) and 3 TF-IDF vectorizers (`tfidf_text`, `tfidf_title`, `tfidf_skills`), plus the 73-skill vocabulary and canonical role mappings, are loaded **once** at startup using `@st.cache_resource`.

### 2. CV Extraction
When a user uploads a PDF CV:
1. `cv_extractor.py` extracts text using `pdfplumber` (falling back to `pypdf`).
2. Regex and NLP rules extract:
   - **Role**: canonicalized using TF-IDF cosine against `KNOWN_ROLES`.
   - **Seniority**: keywords (`Senior`, `Lead`, `Junior`, etc.) or years mapping.
   - **Years of Experience**: parsed from explicit patterns and date ranges.
   - **Education**: PhD, MBA, MSc, BSc, BA, or High School.
   - **Skills**: extracted against `SKILL_VOCAB` using `SKILL_ALIASES` (e.g. `node.js` → `JavaScript`, `postgresql` → `Databases`).
3. The candidate profile is validated before processing.

### 3. Arbeitnow Job Board Integration
1. `job_api.py` retrieves live jobs from `https://www.arbeitnow.com/api/job-board-api` with timeout and error handling.
2. Results are cached with `@st.cache_data(ttl=600)` (10 minutes) to prevent redundant API queries.
3. `job_cleaner.py` removes HTML tags (`<b>`, `<p>`, `<li>`), decodes entities, removes bare URLs and markdown, while strictly preserving metadata and original application links.
4. Skills are parsed from both description and tags against `SKILL_VOCAB`.

### 4. Feature Engineering
`feature_engineering.py` implements the exact 19 features from notebook cell 87:
- **Skill features (8)**: `n_must_have`, `n_matching_skills`, `n_missing_skills`, `must_have_coverage`, `n_nice_matching`, `skill_jaccard`, `n_candidate_skills`, `skill_tfidf_sim`.
- **Non-skill features (11)**: `years_experience`, `exp_in_range`, `exp_gap`, `candidate_seniority`, `job_seniority`, `seniority_match`, `seniority_diff`, `role_match`, `industry_match`, `title_sim`, `text_sim`.

### 5. Prediction & Confidence
- The model outputs `prediction` (1 = Match, 0 = No Match).
- Confidence is obtained from `model.predict_proba(X)[0, 1]` (positive class probability).
- Results are ranked strictly in descending order of model match confidence.
- **Important**: The application clearly labels this as **model candidate-job compatibility confidence**, NOT guaranteed hiring probability.

### 6. Shared and Missing Skills
Computed using case-insensitive normalization while returning canonical display forms from `SKILL_VOCAB`:
- `shared_skills`: candidate skills required by the job.
- `missing_skills`: required skills the candidate does not currently possess.

---

## 🧪 Testing

Run the comprehensive unit and validation test suite:

```bash
python test_suite.py
```

Run the end-to-end integration test with live Arbeitnow jobs:

```bash
python test_end_to_end.py
```
