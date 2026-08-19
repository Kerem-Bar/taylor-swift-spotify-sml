# <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="30" height="30" valign="middle"> Predicting Spotify Track Popularity: Supervised Regression Modeling of Audio and Structural Features

An empirical Supervised Machine Learning (SML) research pipeline developing predictive regression architectures to forecast track popularity from combined acoustic-structural features and assessing their broader industry generalizability. This repository benchmarks parametric Ordinary Least Squares (OLS) Linear Regression against non-parametric tree ensembles (Random Forest and Gradient Boosting Regressors).
## 📌 Project Overview & Feature Architecture

Traditional music popularity forecasting frequently falls into an "Acoustic Isolation Trap", predicting commercial success strictly from raw audio signals while ignoring chronological and structural album context. To bridge this empirical gap, this study formulates track popularity forecasting (*Y* ∈ [0, 100]) as a supervised machine learning regression task over Taylor Swift's extensive discography (*N* = 579), utilizing her diverse two-decade career and stylistic shifts as an optimal case study.

The predictive framework models track popularity against an independent feature matrix (*X*) of 12 numeric predictors partitioned into two functional categories:

* **Nine Intrinsic Acoustic Features:** `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, and `valence`.
* **Three Structural-Conceptual Features:**
  * `track_number`: Album sequential position.
  * `duration_ms`: Track length in milliseconds.
  * `release_year`: Calendar year of release.
## 🔬 Core Research Questions

* **RQ1 (Predictive Capacity):** Can supervised machine learning architectures accurately predict individual song popularity based on a combined acoustic-conceptual feature matrix?
* **RQ2 (Generalizability):** To what extent can an artist-specific predictive framework successfully generalize across the broader music industry?





---

## 📁 Repository Structure

```text
taylor-swift-spotify-sml/
│
├── Code & Data Execution Guide.pdf # Step-by-step Execution & Reproduction Guide
├── Taylor_swift_smlproject.py       # Initial Data Science Lifecycle, Exploratory Baselines & CSV Export
├── winning_models_sml2.py           # Final Production Script (OLS, Random Forest & Gradient Boosting)
├── taylor_swift_spotify.csv         # Raw Spotify Dataset (582 records, 18 columns)
├── taylor_baseline.csv              # Preprocessed Dataset tailored for OLS (11 features, N = 579)
├── taylor_improved.csv              # Preprocessed Dataset tailored for Tree Ensembles (12 features, N = 579)
├── SML final project Kerem Bar.pdf  # Comprehensive SML Academic Research Report
├── Figure1.png                      # Boxplot Outlier Isolation for Target Variable Popularity
├── Figure2.png                      # Song Popularity Distribution by Album Families
├── Figure3.png                      # Standardized Linear Coefficient Weights
├── Figure4.png                      # Feature Importance of Both Tree Ensemble Models
├── Figure5.png                      # Empirical Evidence of Predictive Variance Collapse
└── README.md                        # Detailed Repository Architecture & Research Findings
```
---

## 📌 Data Preprocessing & Leak-Free Architecture

To establish a stable and mathematically robust predictive environment across a multi-era discography, the preprocessing pipeline executes four structural engineering steps:

### 1. Outlier Isolation & Dataset Finalization (Figure 1)

Exploratory IQR analysis on the raw dataset (*N* = 582) established a mathematical lower fence at 7.5 popularity points. Exactly 3 observations fell below this threshold with a continuous popularity score of 0. Targeted qualitative inspection confirmed these entries were non-musical voice memos from the *1989 (Deluxe Edition)*. To eliminate structural noise and stabilize modeling variance, a deterministic filter (`popularity > 0`) was applied, finalizing a clean modeling space of *N* = 579 musical tracks.

### 2. Catalog Structuring & Leak-Free Group Partitioning (Figure 2)

To resolve severe data leakage caused by sonically identical entries across catalog re-recordings and expanded editions, individual release titles were consolidated into unified `album_family` entities (for example, consolidating the original, deluxe, and re-recorded releases of *1989* under a single `1989` album family). 

The discography was partitioned using `scikit-learn`'s `GroupShuffleSplit` under an out-of-sample holdout strategy (80/20 train-test ratio). This guaranteed that entire unseen eras, comprising the `1989`, `folklore`, and `evermore` album families, were strictly isolated into `X_test`, ensuring the models were evaluated entirely on out-of-sample discographic contexts. This exact train-test partition (`X_train`, `X_test`, `y_train`, `y_test`) was held identical across all three benchmarked regression architectures to ensure strict comparative validity.

### 3. Chronological Feature Extraction & Categorical Dummy Substitution

* **Temporal Feature Extraction (`release_date` → `release_year`):** Extracted four-digit calendar years from raw date strings to represent chronological progression as a continuous integer feature.
* **Continuous Proxies over Categorical Dummies:** Applying One-Hot Encoding directly to `album_family` under group partitioning creates zero-variance dummy columns in `X_train` (all zeros), preventing valid OLS coefficient estimation and tree splits. To preserve album context without categorical breakdown, continuous structural features (`release_year`, `track_number`, and `duration_ms`) were utilized as numeric proxies for era, album pacing, and compositional scale.



### 4. Leak-Free Feature Standardization

Following the dataset split, Z-score standardization (`StandardScaler`) was applied across the modeling pipeline to align features with vastly different scales (such as `duration_ms` versus audio metrics bounded between 0 and 1). To strictly prevent data leakage, the scaler was fitted exclusively on `X_train`, and then used to transform both `X_train` and `X_test`.

---

## 📊 Model Evaluation & Performance Benchmarking

### 🔍 Multicollinearity Diagnostics & Energy Sensitivity Analysis

Empirical evaluation via Pearson correlation identified a severe linear dependency between `energy` and `loudness` (r = 0.80). In Ordinary Least Squares (OLS), this multicollinearity inflates coefficient variance and destabilizes slope estimation (β). To isolate this effect, the baseline was evaluated under two distinct specifications:

* **Linear Regression (Full):** Incorporates all 12 predictors (includes `energy`) to diagnose coefficient instability.
* **Linear Regression (Stabilized):** Excludes `energy` to resolve collinearity and restore reliable slope estimation.

Omitting `energy` yielded slight out-of-sample improvements across both Test RMSE and Test R², establishing the stabilized configuration as the preferred linear baseline for benchmarking against the tree ensembles.
### 🏆 Empirical Benchmark Results & Specification Comparison

Across the three core regression architectures, four experimental specifications were evaluated across identical partition splits (`X_train` vs. `X_test`). Hyperparameter tuning via grid search enforced a maximum tree depth boundary of 4 across tree ensembles to control model complexity and mitigate overfitting.
| Model Architecture | Feature Configuration | Train RMSE | Test RMSE | Train R² | Test R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Full)** | Full 12 Features | 11.4240 | 10.8603 | 0.5595 | -0.1536 |
| **Linear Regression (Stabilized)** | 11 Features (Excludes `energy`) | 11.4302 | 10.7815 | 0.5590 | -0.1369 |
| **Random Forest Regressor** | Full 12 Features | 7.6028 | 10.7317 | 0.8049 | -0.1264 |
| **Gradient Boosting Regressor** | Full 12 Features | 3.2835 | 10.7190 | 0.9636 | -0.1238 |
---

## 🔍 Failure Diagnosis & Empirical Breakdown

### 1. Linear Model Diagnosis: Rigid Acoustic Rules & Chronological Distortion (Figure 3)

The out-of-sample breakdown of the OLS baseline demonstrates that a uniform acoustic formula for popularity does not exist across a multi-era catalog:
* **The Flawed Acousticness Rule (+1.2395):** The model established a rigid positive assumption equating high acoustic density with commercial success. While this rule held true for prominent frontline hits (popularity 77–78 at acousticness 0.77–0.92), it collapsed on deeper acoustic cuts where near-absolute acousticness (0.964) dropped to commercial lows (popularity 43), exposing internal catalog polarization.
* **The Flawed Liveness Rule (-0.7238):** The fixed negative penalty failed to distinguish maximalist studio productions (which registered elevated liveness 0.32–0.38 due to layered arena reverberation yet achieved peak popularity 71–80) from raw live sessions (liveness 0.791, popularity 54).
* **Over-Indexed Chronology (`release_year` = +12.2020):** Driven by a massive positive coefficient of +12.2020 assigned to `release_year`, the model severely penalized earlier releases of sonically identical tracks, creating artificial valuation gaps between the 2014 original and 2023 re-recording of `1989`.

### 2. Tree Ensemble Diagnosis: The Chronological Trap & Predictive Variance Collapse (Figures 4 & 5)

Expanding algorithmic capacity to non-linear tree ensembles failed to resolve this predictive ceiling due to two structural mechanics:
* **The Chronological Trap:** `release_year` completely dominated tree splitting hierarchies, capturing 70.4% feature importance in Random Forest and 69.6% in Gradient Boosting (with all remaining features failing to clear an 8% threshold). Mirroring the linear baseline, the ensembles operated as macro-historical binning systems rather than musical estimators, splitting identically produced tracks based strictly on calendar year (2014 vs. 2023).
* **Predictive Variance Collapse:** By prioritizing chronological bins over acoustic nuance, the models compressed true empirical variance into near-static predictions. Most notably on *folklore* (2020), an actual standard deviation of σ = 11.23 collapsed to a rigid predicted spread of just σ = 0.53 in Random Forest and σ = 2.71 in Gradient Boosting (Figure 5).
### 3. Summary of Empirical Breakdown

Across both parametric and non-parametric architectures, applying rigid assumptions to skewed feature spaces produced out-of-sample prediction errors where Residual Sum of Squares strictly exceeded Total Sum of Squares (SSres > SStot). This structural limitation caused all evaluated models to collapse below zero (Test R² < 0), performing worse than a naive benchmark predicting the mean of `y_test`.









---

## 🎯 Key Conclusions & Future Work

### 📌 Research Question Findings

Based on empirical evaluations across the catalog, both research questions yielded a definitive negative outcome:

* **RQ1 Conclusion (Negative) - Track Popularity Prediction:** Individual song popularity cannot be predicted using acoustic-conceptual feature matrices alone. Rather than learning true musical patterns, the models relied on `release_year` as a macro chronological shortcut, failing to evaluate out-of-sample eras and collapsing to negative predictive performance (R² < 0).
* **RQ2 Conclusion (Negative) - Industry Generalizability:** The predictive framework cannot generalize across the broader music industry. Because the decision logic is strictly tied to Taylor Swift's unique career timeline and catalog re-recording milestones, the architecture functions as an artist-specific career tracker rather than a transferable musical estimator.

### 🚀 Proposed Future Directions

To overcome these structural boundaries while preserving catalog integrity, future iterations should implement two practical methodological adjustments:

* **Distribution-Based Preprocessing:** Apply a standard `log(x + 1)` transformation exclusively to highly skewed acoustic parameters (e.g., `speechiness`, `instrumentalness`) to stabilize training leverage and accommodate zero-value bounds.
* **Chronological Abstraction via External Exposure Proxies:** Replace the static, artist-specific `release_year` feature with dynamic, time-varying external signals (such as media appearance frequencies, social media engagement volume, and tour scheduling intensity) to capture promotional momentum without falling into the chronological trap.
---

## 📊 Visualizations Highlights

| <div style="width: 280px">Figure 1: Boxplot Outlier Isolation</div> | Figure 2: Popularity by Album Family |
| :---: | :---: |
| <img src="Figure1.png" width="100%"> | <img src="Figure2.png" width="80%"> |
| *Box plot isolating 3 voice memo outliers with 0 popularity.* | *Box plot illustrating popularity dispersion across album families.* |

| Figure 3: Standardized OLS Coefficients | Figure 4: Feature Importance Comparison |
| :---: | :---: |
| <img src="Figure3.png" width="90%"> | <img src="Figure4.png" width="90%"> |
| *Feature weights in stabilized OLS (+12.20 release_year).* | *Feature importance across RF and GB (~70% release_year).* |

| Figure 5: Predictive Variance Collapse |
| :---: |
| <img src="Figure5.png" width="50%"> |
| *Empirical proof comparing actual song variance against flattened predictions in X_test.* |
---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Science Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization Tools:** Matplotlib, Seaborn
* **Primary Models:** `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`
* **Validation Modules:** `GroupShuffleSplit`, `StandardScaler`

---

## 🚀 How to Execute the Pipeline

For detailed execution paths, dependencies, and environment configuration, refer to the included `Code & Data Execution Guide.pdf`.

The project follows a two-stage modular execution architecture:

* **Stage 1 — Exploratory Lifecycle & Preprocessing (`Taylor_swift_smlproject.py`):** Ingests the raw Spotify dataset (`taylor_swift_spotify.csv`), executes outlier filtering and feature engineering, exports `taylor_baseline.csv` and `taylor_improved.csv`, and generates exploratory visualizations (**Figures 1 & 2**).
* **Stage 2 — Benchmark & Evaluation (`winning_models_sml2.py`):** Ingests the preprocessed CSVs, trains the final benchmark models (OLS, Random Forest, Gradient Boosting), outputs performance metrics, and generates diagnostic evaluation plots (**Figures 3, 4, & 5**).

```bash
# 1. Clone the repository
git clone [https://github.com/Kerem-Bar/taylor-swift-spotify-sml.git](https://github.com/Kerem-Bar/taylor-swift-spotify-sml.git)

# 2. Stage 1: Data preprocessing & Figures 1-2
python Taylor_swift_smlproject.py

# 3. Stage 2: Model training & Figures 3-5
python winning_models_sml2.py
```

---

## 👤 Author

**Kerem Bar**  
Master's Student in Information Sciences (Information Technology Specialization)
