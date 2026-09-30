# Obesity Risk Insurance ML

A machine learning pipeline for predicting obesity risk classification, built for the **Modeling the Future Challenge (MTFC)**, a national actuarial modeling competition. The project frames obesity risk as an insurance underwriting problem: predicting an individual's weight-status category from lifestyle and biometric features, then using the trained models to simulate long-term risk trajectories for pricing and mitigation analysis.

The work accompanies the team paper *The Obesity Epidemic: A Novel Analysis of Patient-Level Obesity and Machine Learning Risk Mitigation Optimization* (Team 20354, "Galaxy Munchkins", March 2025).

This was a 5-person team project. I was responsible for the full modeling and programming analysis, which is what's contained in this repository.

## Project context

This project was originally developed in 2024–2025 for the Modeling the Future Challenge. This repository is a cleaned archival version created to share the modeling code, documentation, and some key results.

## Pipeline

![Pipeline overview](results/pipeline_overview.png)

Raw survey/biometric data is cleaned and feature-engineered, then passed through several classification models (with Bayesian hyperparameter optimization), evaluated against each other, and the best-performing model (XGBoost) feeds into a per-sample Monte Carlo simulation that recommends lifestyle changes. Those recommendations are then applied to a 20-year Markov chain to project long-term population and insurance-cost impact.

## Key Results

| Model | Test accuracy |
|---|---|
| XGBoost (TPE-tuned) | 85.82% |
| FT-Transformer (TPE-tuned) | 82.00% |
| Multinomial logistic regression (baseline) | 53.00% |

- **Obese individuals:** the Monte Carlo intervention reduced obesity probability by **36.14%** (95% CI 33.14-39.14%).
- **Overweight individuals:** overweight probability fell **13.5%** (95% CI 9.8-17.19%), and obesity probability fell from 10.5% to 7.9% (24.8% decrease).
- **Sensitivity analysis:** snacking between meals (CAEC) had the largest effect on overweight/obese classification; physical activity (FAF) and water intake (CH2O) had more gradual effects, and coordinated changes amplified the shifts.
- **Long-term impact (100,000 hypothetical policyholders, 20 years):** obesity prevalence falls from 43.35% to 31.69% under the intervention-adjusted Markov chain, a 26.90% relative reduction. Estimated annual claims cost drops from about $67.9M to $52.9M, saving roughly $15.0M.

## Methodology

**Phase 1: patient-level classification.** XGBoost and FT-Transformer classifiers predict one of seven weight classes from the UCI dataset (2,111 samples, 17 features). `Weight` is dropped because it directly defines the BMI-based target, and `Gender` is dropped to avoid bias from the class imbalance across genders. Categorical features are label-encoded and continuous features are Z-scored. Hyperparameters are tuned with Optuna's Tree-structured Parzen Estimator (100 trials for XGBoost using 5-fold CV, 50 trials with pruning for FT-Transformer).

**Phase 2: per-sample Monte Carlo risk mitigation.** For each obese or overweight individual, 500 candidate interventions are sampled (deltas drawn from N(0, 0.2), constrained in the healthy direction) over four modifiable features: physical activity (FAF), calorie monitoring (SCC), snacking between meals (CAEC), and daily water intake (CH2O). The trained XGBoost model scores each candidate, and the one minimizing a loss is chosen. The loss is the log of the target-class probability, a penalty for pushing underweight probability above a threshold (and, for overweight individuals, for increasing obesity probability), and an L2 penalty on the size of the change. Before/after probability distributions are compared with kernel density estimates.

**Phase 3: Markov chain.** A four-state discrete-time Markov chain (Insufficient, Normal, Overweight, Obese) is fit to NHANES 1988-2018 weight-class prevalence via constrained optimization to get a yearly baseline transition matrix. The intervention-adjusted matrix applies element-wise multipliers derived from the Monte Carlo results, is row-normalized, and is iterated for 20 years starting from the current US weight distribution.

**Assumptions:** the two overweight levels split the CDC 25-29.9 BMI range at 27.5; the transition matrix is constant year to year; NHANES has no insufficient-weight column, so it is assumed to decline linearly from 2.3% (1988) to 1.6% (2018); NHANES trends carry to 2025; policyholder count is constant; the synthetic UCI data is representative of real-world obesity patterns.

**Data caveat:** about 77% of the UCI dataset was generated synthetically with Weka/SMOTE (23% from real survey responses in Mexico, Peru, and Colombia), so model accuracy should be read with that in mind.

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/obesity-preprocessing.ipynb` | Data cleaning, feature engineering, and exploratory analysis |
| `notebooks/LogisticRegression.ipynb` | Baseline logistic regression classifier |
| `notebooks/XGBoost.ipynb` | XGBoost classifier |
| `notebooks/XGBoost-labelencoding-and-Monte-Carlo.ipynb` | XGBoost variant with label-encoded categorical features, plus the per-sample Monte Carlo risk mitigation simulation |
| `notebooks/xgb_oat_and_lhs.ipynb` | One-at-a-time and Latin hypercube sampling sensitivity analysis on the XGBoost model |
| `notebooks/FTTransformer.ipynb` | Feature Tokenizer Transformer (FT-Transformer) deep learning classifier |
| `notebooks/SAINT.ipynb` | SAINT (Self-Attention and Intersample Attention Transformer) classifier |
| `notebooks/Markov_chain.ipynb` | 20-year Markov chain simulation of baseline vs. intervention-adjusted weight-class trajectories |

Models were tuned using Tree-structured Parzen Estimator (TPE) Bayesian optimization and evaluated by classification report, ROC-AUC (one-vs-rest), and confusion matrix across seven weight-status classes (Insufficient Weight, Normal Weight, Overweight Level I/II, Obesity Type I/II/III).

## Results

**Exploratory data analysis**

![Class distribution](results/class_distribution.png)
![UCI dataset overview](results/UCI_selected_4_plots.png)

**Baseline (Logistic Regression)**

![Baseline classification report](results/baseline_classification_report.png)

**XGBoost (tuned)**

![XGBoost hyperparameter search](results/optimal_hyperparameters_xgb.png)
![XGBoost classification report](results/optimized_classification_report_xgb.png)
![XGBoost ROC curve](results/optimized_xgb_roc_ovr.png)
![XGBoost feature importances](results/xgb_final_feature_importances.png)

**FT-Transformer**

![FT-Transformer architecture](results/fttransformer_architecture.png)
![FT-Transformer hyperparameter search](results/optimal_hyperparameters_fttransformer.png)
![FT-Transformer classification report](results/fttransformer_classification_report.png)
![FT-Transformer ROC curve](results/fttransformer_roc_curve.png)

**Monte Carlo and Markov Chain risk simulations**

![Markov chain diagram](results/markov.png)
![Obesity probability distribution](results/obesity_probabilities_distribution_MC.png)
![Sensitivity analysis](results/sensitivity_analysis_weight_categories.png)
![Obesity type comparison](results/obesity_type_comparison.png)

## Product Recommendations

The models feed a concrete insurance product design, not just predictions:

- **Personalized lifestyle plan (outcomes modification).** Each policyholder gets unit-level targets from the Monte Carlo optimizer. Obese: physical activity +0.15-0.78, daily calorie tracking (SCC = 1), snacking -0.15 to -0.33, water +0.15-0.63. Overweight: activity +0.15-0.61, calorie tracking, snacking -0.15 to -0.22, water +0.15-0.64. In survey terms this means, for example, moving from exercising 1-2 days a week to 2-4, or from under 1 L of water a day to 1-2 L.
- **Tiered premium discounts** on a $500 base annual premium: 5% (baseline, on enrollment, $475), 10% (active engagement, $450), 15% (optimal health after 12 months of consistent results, $425).
- **Cashback:** $20/month when combined wellness targets are met, plus a $150 annual bonus for optimal health performance.
- **Coverage for obesity-related conditions** (diabetes, hypertension, cardiovascular disease, certain cancers): 70% standard, up to 80% for sustained improvement, up to 85% with an annual performance bonus.
- **Implementation:** a required digital health platform for continuous monitoring, regular progress assessments, and quarterly reports to policyholders on discount status, rewards, and coverage.
- **Population-level behavior change** (schools, parks, bike lanes, recreation centers) was recommended as complementary prevention, separate from the underwriting model.

## Financial Impact

Modeled for a hypothetical insurer with 100,000 policyholders representative of the US weight distribution, assuming 70% coverage of weight-related medical costs and excess annual costs of $1,816 per obese and $600 per overweight adult (CDC/NCHS-sourced figures cited in the paper).

| Year-20 scenario | Obese | Overweight | Annual claims expense |
|---|---|---|---|
| Baseline | 43,524 | 30,037 | $67,943,249 |
| With interventions | 31,689 | 30,037 | $52,898,597 |
| **Savings** | | | **$15,044,652** |

Over the 20-year Markov simulation, the intervention-adjusted population also keeps roughly 6,000 more people in the normal-weight range (30,943 vs. 24,931) and cuts obesity prevalence from 43.35% to 31.69%. For scale, the baseline severity estimate for the current population is about $95.4M per year.

## Limitations

- The savings figure holds the overweight count at the baseline value (30,037) in both scenarios. The intervention scenario actually ends with 35,109 overweight policyholders; recomputing with that count gives about $55.0M in expense and roughly $12.9M in savings (my recalculation, not from the paper). Either way the direction and rough size of the result hold.
- Savings are gross: they do not net out discounts, cashback, coverage upgrades, or platform costs, so this is not a full pricing or profitability model.
- The Monte Carlo interventions are model-predicted changes in class probability, not observed behavior change. Real adherence and causal effects were not measured.
- Roughly 77% of the training data is synthetic (SMOTE), and the Markov transition matrix is assumed constant and is calibrated to NHANES aggregates, not individual trajectories.
- This was a competition project. The insurance product was designed and simulated but never deployed or piloted.

## End-to-End Scope

| Stage | What was built |
|---|---|
| Problem framing | Obesity risk recast as an insurance underwriting and cost-mitigation problem |
| Data | Cleaning, feature selection (dropped leakage-prone `Weight` and bias-prone `Gender`), NHANES calibration |
| Modeling | Three classifiers benchmarked against a baseline, with Bayesian hyperparameter tuning |
| Decision layer | Per-person Monte Carlo optimization that turns predictions into actionable recommendations |
| Long-term projection | Markov chain projecting population impact over 20 years |
| Product and business case | Tiered incentive and coverage design with a quantified expense comparison |
| Communication | 30-page paper and this repository |

## Data

This repository does not include the underlying datasets or trained model weights (to keep the repo lightweight and avoid redistributing third-party data). The project draws on publicly available data from:

- [UCI Machine Learning Repository: Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- CDC Behavioral Risk Factor Surveillance System (BRFSS)
- NHANES (National Health and Nutrition Examination Survey)

To reproduce results, download the relevant dataset(s) above and point the preprocessing notebook at them.

## Stack

Python, pandas, scikit-learn, XGBoost, PyTorch, Optuna/TPE (hyperparameter optimization), Matplotlib.

## Disclaimer

This project was created for educational and actuarial modeling purposes as part of the Modeling the Future Challenge. It is not a medical diagnostic tool, clinical recommendation system, insurance underwriting system, or financial product. Model outputs and simulations are intended only to demonstrate machine learning and risk-modeling methods on public datasets, and should not be used to make real-world medical, insurance, or financial decisions.
