# Airline Merger Damages Estimator

**Causal inference analysis of consumer harm from the 2013 American Airlines / US Airways merger**

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-red)](https://merger-app-1010508609944.us-central1.run.app)
[![API](https://img.shields.io/badge/API-Flask%20%7C%20Cloud%20Run-blue)](https://merger-api-1010508609944.us-central1.run.app/health)

---

## Live Services

| Service | URL |
|---|---|
| Streamlit App | https://merger-app-1010508609944.us-central1.run.app |
| Flask API | https://merger-api-1010508609944.us-central1.run.app |
| API Health Check | https://merger-api-1010508609944.us-central1.run.app/health |

---

## Project Overview

This project replicates the **antitrust damages estimation workflow** used by economic consulting firms (Analysis Group, Compass Lexecon, NERA) in litigation. When the December 2013 merger between American Airlines and US Airways reduced direct competition on over 1,200 domestic routes, it created a natural experiment for causal inference.

**Core research questions:**
- Did the merger causally raise ticket prices on routes where AA and US Airways previously competed?
- By how much per ticket, and what is the total dollar value of consumer harm?
- Which routes and markets were harmed most?

**Key finding:** The merger caused a statistically significant **2.57% fare increase** (95% CI: 1.13% to 5.44%) on treated routes, corresponding to **$3.97 billion in total consumer harm** across 622 million affected passengers from 2014-2017.

A critical methodological finding is the **sign flip**: standard linear estimators (OLS, Two-Way Fixed Effects) produce a spurious *negative* treatment effect because they cannot flexibly absorb the nonlinear confounding of the 2014-2015 oil price crash. Double ML with LightGBM correctly resolves this.

---

## Solution Architecture

```
Data Sources
    |
    +-- DOT BTS DB1B API ──────────────────+
    |   (32 quarters, 2010-2017)           |
    +-- FRED API (jet fuel, airline CPI) --+
                                           |
                                           v
                                 Data Pipeline (Python)
                                 download_data.py
                                           |
                                           v
                              Processed Panel Dataset
                              122,787 route-quarter obs
                                           |
                                           v
                                  Model Training
                              (notebooks/03_modeling.ipynb)
                          OLS -> TWFE -> Double ML -> Causal Forest
                                           |
                                           v
                              Model Artifacts (joblib .pkl)
                              damages_summary.parquet
                                           |
                                           v
                         +----------------+----------------+
                         |                                 |
                         v                                 v
              Flask API (Cloud Run)             Streamlit App (Cloud Run)
              /health, /routes,                 Executive Summary
              /damages, /predict                Route Analysis
                                                Methodology
                                                         |
                                                         v
                                                     End User
```

---

## Repository Structure

```
stat418-airline-merger-damages/
|
+-- README.md                        <- This file
+-- download_data.py                 <- Automated data collection script
+-- CLAUDE_CONTEXT.md               <- AI assistant briefing document
+-- .gitignore
|
+-- notebooks/
|   +-- 01_eda.ipynb                <- Exploratory data analysis
|   +-- 02_preprocessing.ipynb      <- Panel construction, feature engineering
|   +-- 03_modeling.ipynb           <- Full modeling pipeline + damages calculation
|   +-- proposal_eda.ipynb          <- Proposal presentation figures
|   +-- db1b_validation.ipynb       <- Initial data validation
|
+-- api/
|   +-- app.py                      <- Flask API (4 endpoints)
|   +-- requirements.txt
|   +-- Dockerfile
|
+-- streamlit_app/
|   +-- app.py                      <- Streamlit dashboard (3 pages)
|   +-- requirements.txt
|   +-- Dockerfile
|
+-- models/
|   +-- artifacts/
|       +-- model_comparison.csv    <- Four-model comparison table
|
+-- docs/
|   +-- figures/                    <- All EDA and modeling plots
|
+-- tests/
|   +-- test_api.py                 <- pytest unit tests for all API endpoints
|
+-- data/
    +-- log/                        <- Download logs
```

---

## Data Collection

Data was collected programmatically from two sources:

### Primary: DOT Bureau of Transportation Statistics DB1B
- **What:** 10% random sample of all domestic airline passenger tickets
- **Method:** Automated bulk download loop — no API key required
- **Coverage:** 32 quarters (2010 Q1 through 2017 Q4)
- **Scale:** ~5 million ticket records per quarter, 4 GB raw data
- **URL pattern:** `https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.zip`

```bash
python download_data.py --fred-key YOUR_KEY_HERE
```

### Control Variables: FRED API
- Jet fuel prices: `WJFUELUSGULF` (weekly, aggregated to quarterly)
- Airline CPI: `CUSR0000SETG01` (monthly, aggregated to quarterly)
- Free API key at: https://fred.stlouisfed.org/docs/api/api_key.html

---

## Modeling Approach

### Identification Strategy
Treatment/control design exploiting the merger as a natural experiment:
- **Treated routes (1,223):** Origin-destination pairs where both AA and US Airways operated pre-merger. These faced direct competitive harm.
- **Control routes (3,721):** Routes where only one carrier operated pre-merger. Used as the counterfactual price trend.
- **Study period:** 2010-2013 (pre-merger) and 2014-2017 (post-merger)

### Model Progression

| Model | ATE | Interpretation |
|---|---|---|
| Pooled OLS | -2.73% | Spurious negative (fuel price confound) |
| Two-Way Fixed Effects | -2.54% | Spurious negative (nonlinear confound not absorbed) |
| **Double ML (PRIMARY)** | **+2.57%** | **Statistically significant fare increase** |
| Causal Forest (mean ITE) | +4.48% | Heterogeneous effects by route |

**Why Double ML?** Time fixed effects absorb economy-wide shocks uniformly — but the 2014-2015 jet fuel price crash affected routes differently depending on distance, competition level, and carrier mix. A linear model cannot capture those interactions, causing the fuel-driven fare decrease to leak into the treatment estimate. Double ML uses LightGBM gradient boosting to partial out all controls flexibly before estimating the causal effect.

### Double ML Procedure (Chernozhukov et al., 2018)
1. Residualize log fare on all controls using LightGBM: `E[log_fare | X]`
2. Residualize the merger treatment indicator on all controls: `E[treated_post | X]`
3. Regress outcome residuals on treatment residuals to recover the causal ATE

### Damages Calculation (But-For Methodology)
Standard approach in antitrust litigation:
- **Counterfactual fare** = Actual fare / (1 + 0.0257)
- **Overcharge per ticket** = Actual fare - Counterfactual fare
- **Total consumer harm** = (Actual fare - Counterfactual fare) x Passengers

**Results:** $3.97B total consumer harm | 622M affected passengers | $6.73 avg overcharge per ticket

---

## API Endpoints

The Flask API is deployed on Google Cloud Run at:
`https://merger-api-1010508609944.us-central1.run.app`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check, confirms artifacts loaded |
| `/routes` | GET | List of 200 treated routes sorted by consumer harm |
| `/damages` | GET | Aggregate damages summary, model comparison table |
| `/predict` | POST | Route-specific overcharge and damages estimate |

**Example predict request:**
```bash
curl -X POST https://merger-api-1010508609944.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"route": "LAX-JFK"}'
```

**Example response:**
```json
{
  "route": "LAX-JFK",
  "found_in_damages": true,
  "avg_fare_actual": 468.01,
  "avg_fare_but_for": 456.27,
  "avg_overcharge": 11.74,
  "total_damages": 58401373.88,
  "total_passengers": 4961880.0
}
```

---

## Streamlit App

Three-page consulting-style dashboard deployed on Google Cloud Run:

**Executive Summary** — Headline metrics (2.57% overcharge, $3.97B total harm, 622M passengers), interactive bar chart of top 20 routes by consumer harm, model comparison table with sign flip explanation, full ranked table of all treated routes.

**Route Analysis** — Route selector dropdown (200 treated routes), actual vs. counterfactual fare metrics, fare comparison bar chart, and fare history line chart (2010-2017) with merger date marker and 95% CI band.

**Methodology** — Full causal identification strategy, Double ML explanation, damages calculation formula, data sources table.

---

## Local Setup

### Prerequisites
- Python 3.11
- FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)

### Installation

```bash
git clone https://github.com/rnarasayya/stat418-airline-merger-damages.git
cd stat418-airline-merger-damages

# Download data (takes ~60 minutes for full 32 quarters)
pip install requests pandas pyarrow
python download_data.py --fred-key YOUR_KEY_HERE

# Install API dependencies
cd api
pip install -r requirements.txt
python app.py  # runs on port 8080

# In a second terminal, install and run Streamlit
cd streamlit_app
pip install -r requirements.txt
# Windows:
set API_URL=http://127.0.0.1:8080
streamlit run app.py
# Mac/Linux:
API_URL=http://127.0.0.1:8080 streamlit run app.py
```

### Running Tests
```bash
pip install pytest
ARTIFACT_DIR=models/artifacts pytest tests/test_api.py -v
```

---

## AI Assistant Documentation

This project was built using multiple AI coding assistants throughout the development process.

### Tools Used

| Tool | Purpose |
|---|---|
| **Claude (claude.ai)** | Specification writing, debugging, code review |
| **Claude Code** | Notebook generation, Flask API, Streamlit app, Dockerfile creation |

### How AI Assistance Was Used

**Architecture and Methodology Design (Claude chat)**
The overall project structure, and causal identification strategy were my own original ideas refined through Claude. From there, the modeling approach (Double ML + Causal Forest), damages calculation methodology, and Flask API design were all developed through iterative conversation with Claude. Prompts focused on explaining the economics consulting context and getting technically rigorous specifications before any code was written.

**Notebook Generation (Claude Code)**
All four Jupyter notebooks were generated by Claude Code using detailed specifications from `CLAUDE_CONTEXT.md` — a living briefing document maintained throughout the project. The specification included exact column names, data cleaning rules, modeling choices, and validation checks so Claude Code could produce complete notebooks in one shot.

Example prompt pattern used:
> "Read CLAUDE_CONTEXT.md in full. Build the preprocessing notebook as specified in the 'Preprocessing Notebook Detailed Specification' section. Follow the memory strategy, cleaning filters, feature engineering steps, HHI computation, FRED merge, validation checks, and save logic exactly as described. All column names and variable conventions must match the spec precisely since the modeling notebook will depend on them."

**Flask API and Streamlit App (Claude Code)**
Both applications were generated by Claude Code from detailed specs in `CLAUDE_CONTEXT.md`. The context file was updated after each stage to reflect confirmed results and carry forward key numbers (e.g., ATE = 2.57%, total damages = $3.97B).

**Debugging (Claude chat)**
Several real issues were diagnosed and fixed through Claude:
- EconML `PopulationSummaryResults` API incompatibility: `ate_inference().summary_frame()` does not exist on newer versions; fix was to use `coef__interval()` instead
- Windows cp1252 encoding errors from emoji/unicode in logging statements — all logging switched to plain ASCII
- FRED series ID `WJETT` does not exist; correct series is `WJFUELUSGULF`
- Plotly `add_vline()` incompatibility with categorical x-axis — replaced with `add_shape()` + `add_annotation()`
- Docker build context path issues with `--source` deployment on Cloud Run

**Areas Where AI Code Required Significant Modification**
- The Double ML inference section required multiple iterations — Claude Code initially used `ate_inference()` which returned a `PopulationSummaryResults` object lacking standard methods; required manual diagnosis and replacement with `coef__interval()`
- The fare history CI band required correction from `add_vline()` to shape-based approach after runtime error
- Windows environment variable syntax (`set` vs `$env:`) required manual correction in several steps

### Lessons Learned
- Maintaining a structured context document (`CLAUDE_CONTEXT.md`) and updating it after each completed stage dramatically improved Claude Code output quality — it avoided re-generating completed work and had accurate numbers to reference
- Being explicit about constraints (Windows encoding, specific library versions, no emoji) in the context doc prevented recurring errors
- Claude Code works best when given complete specifications rather than open-ended tasks — the more specific the spec, the less iteration was needed
- Separating architecture/design conversations (Claude chat) from code generation (Claude Code) produced better results than trying to do both in one tool

---

## Course Information

**Course:** STAT 418 — Tools in Data Science (UCLA)
**Quarter:** Spring 2026
**Student:** Rohan Narasayya

---

## Acknowledgments

- DOT Bureau of Transportation Statistics for the DB1B public dataset
- Federal Reserve Bank of St. Louis (FRED) for jet fuel and CPI data
- EconML team (Microsoft Research) for the Double ML implementation
- Chernozhukov et al. (2018) for the Double ML methodology
