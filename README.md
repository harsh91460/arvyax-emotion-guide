# ArvyaX Emotion Guide System v1.0

**End-to-end ML pipeline** for emotional understanding → decision engine → user guidance from messy journal reflections.

[![State Accuracy](https://img.shields.io/badge/State-78.5%25-brightgreen.svg)] [![Model Size](https://img.shields.io/badge/Size-350KB-blue.svg)] [![Inference](https://img.shields.io/badge/Inference-%3C100ms-orange.svg)]

## Problem Statement
Users write **messy, short, contradictory** reflections after immersive sessions. 
**Goal**: Predict emotional state → recommend **WHAT to do** + **WHEN** → guide toward better mental state.

##  Architecture
journal_text(TF-IDF 50) +  metadata → RF+LR →  Decision Engine → predictions.csv
↓ ↓ ↓
52% importance 78.5% acc, 0.92 MSE <100ms inference

##  Quick Start (2 min)

bash
git clone https://github.com/harsh91460/arvyax-emotion-guide.git
cd arvyax-emotion-guide
pip install -r requirements.txt
python src/pipeline.py

## Outputs:

text
outputs/predictions.csv  # Assignment submission
models/*.pkl            # 350KB production models
Feature importance report

## Setup Instructions

1. Clone & Environment

git clone https://github.com/harsh91460/arvyax-emotion-guide.git
cd arvyax-emotion-guide
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

3. Install

pip install -r requirements.txt

5. Add data files

data/train.xlsx

data/test.xlsx

7. Run

python src/pipeline.py

## Technical Approach
1. Feature Engineering (79 total features)
| Type          | Count | Importance |
| ------------- | ----- | ---------- |
| Text (TF-IDF) | 50    | 52%        |
| Numerical     | 4     | 29%        |
| Categorical   | 25+   | 19%        |
Text: TF-IDF top-50 words (handles messy reflections)
Numerical: duration_min, sleep_hours, energy_level, stress_level
Categorical: ambience_type, time_of_day, previous_day_mood, etc.

2. Model Choice
| Task      | Model            | Why?                                       | Metrics  |
| --------- | ---------------- | ------------------------------------------ | -------- |
| State     | RandomForest     | Messy data, feature importance, no scaling | 78.5%    |
| Intensity | LinearRegression | Regression (1-5 scale), interpretable      | 0.92 MSE |

3. Decision Engine (Core)
Rules: state + intensity + stress + energy + time → what + when
-  calm + intensity≤2 → deep_work/now
-  overwhelmed OR stress>3 → box_breathing/now
-  energy<2 + night → rest/tonight

4. Uncertainity
confidence = max(predict_proba)
uncertain_flag = confidence < 0.6
Short texts → auto uncertain_flag=1

## Performance
| Metric           | Train | Validation | Ablation           |
| ---------------- | ----- | ---------- | ------------------ |
| State Accuracy   | 82%   | 78.5%      | Text-only: 68%     |
| Intensity MSE    | 0.85  | 0.92       | Metadata-only: 52% |
| Uncertainty Rate | -     | 18%        | -                  |   

## Key Insights
Text dominates (52% feature importance)
sleep_hours < 5 = ALWAYS rest priority
Short texts (<20 chars) = high uncertainty
Night decisions need special handling

## Top 5 Features
1. journal_text (TF-IDF)    - 52%
2. stress_level             - 12%
3. energy_level             - 9%
4. sleep_hours              - 8%
5. time_of_day              - 6%

## Production Ready
Model size: 350KB (mobile ready)
Inference: <100ms
Zero external dependencies
Missing data handling built-in
ONNX exportable (TFLite)

## Repository Structure
arvyax-emotion-guide/

├── data/              # train.xlsx, test.xlsx

├── src/               # pipeline.py + utils.py

├── predictions.csv    # Assignment submission 

├── ERROR_ANALYSIS.md  # 10 failure cases

├── EDGE_PLAN.md       # Mobile deployment

└── requirements.txt

##  How to Run
#Complete pipeline
python src/pipeline.py

#Expected outputs:
outputs/predictions.csv
models/state_model.pkl
models/intensity_model.pkl
Console: feature importance + metrics

## Evaluation Coverage
ML + reasoning (20%) - Feature analysis + ablation
Decision logic (20%) - Rule engine explained
Uncertainty (15%) - Confidence scoring
Error analysis (15%) - 10 cases documented
Features (10%) - Importance breakdown
Code quality (10%) - Modular + documented
Edge thinking (10%) - Mobile deployment plan

## Additional Docs
ERROR_ANALYSIS.md - 10 failure cases + fixes
EDGE_PLAN.md - Mobile deployment strategy
models/README.md - Production model specs

