# <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="30" height="30"> Predicting Spotify Track Popularity: Supervised Regression Modeling of Audio and Structural Features
An empirical Supervised Machine Learning (SML) research pipeline evaluating the predictive capacity of combined acoustic-conceptual feature matrices across Taylor Swift's Spotify catalog. This repository addresses the traditional "Acoustic Isolation Trap" by benchmarking parametric Ordinary Least Squares (OLS) Linear Regression against non-parametric tree ensembles (Random Forest and Gradient Boosting Regressors) under a strict, leak-free validation architecture.

---

## 📌 Project Overview & Research Questions

Traditional commercial music forecasting frequently relies on isolated audio descriptors without contextual positioning. This study formulates track popularity forecasting (Y in range 0 to 100) as a supervised machine learning regression task over Taylor Swift's discography (N = 579). The primary objective is to evaluate whether integrating intrinsic acoustic signals with structural catalog parameters can accurately predict streaming popularity and generalize across the music industry.

### 🔬 Core Research Questions
* **RQ1 (Predictive Capacity):** Can supervised machine learning architectures accurately predict individual song popularity based on a combined acoustic-conceptual feature matrix?
* **RQ2 (Industry Generalizability):** To what extent can an artist-specific predictive framework successfully generalize across the broader music industry?

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

## 🛠️ Data Preprocessing & Leak-Free Architecture

To establish a stable and mathematically sound predictive environment across a multi-era discography, the preprocessing pipeline executed four structural engineering steps:

### 1. Outlier Isolation & Dataset Finalization (Figure 1 & Figure 2)
Exploratory IQR analysis on the raw dataset (N = 582) established a mathematical lower fence at 7.5 popularity points. Exactly 3 records fell below this threshold with a popularity score of 0. Targeted inspection confirmed these entries were non-musical voice memos from 1989 (Deluxe Edition). To eliminate structural noise and stabilize modeling variance, a deterministic filter (`popularity > 0`) was applied, finalizing a clean modeling space of N = 579 musical tracks. Furthermore, visualizing popularity across album families (Figure 2) highlighted contrasting internal popularity profiles ranging from high-popularity compressed tiers (TTPD) to wide vertical dispersion (reputation).

### 2. Chronological Feature Extraction & Conceptual Substitution
To enable text-based release dates for numeric regression estimators, four-character slicing extracted the calendar year into `release_year`. To bypass the curse of dimensionality and zero-variance training constraints caused by categorical dummy indicators in out-of-sample splits, explicit album identities were proactively replaced with continuous structural features (`release_year`, `track_number`, `duration_ms`) to capture temporal and layout context natively.

### 3. Leak-Free GroupSplit Validation Strategy
Standard random splits cause severe representation leakage in music discographies, as original tracks and re-recorded editions share near-identical acoustic footprints. To eliminate cross-partition contamination from repeated tracks across album versions, an out-of-sample `GroupShuffleSplit` (80/20) was executed on engineered `album_families`. 

> **Structural Example:** The engineered **1989 album family** combines the original 2014 release, the 2014 Deluxe edition, and the 2023 Taylor's Version re-recording. A standard random split would scatter these identical acoustic signatures across training and testing partitions. Applying an album-family GroupSplit completely isolated three full musical eras into the holdout test set (X_test): 1989 (comprising 2014 original and 2023 re-recording), folklore (2020), and evermore (comprising 2020 original and 2021 deluxe version).

### 4. Feature Standardization & Multicollinearity Management
Following partitioning, Z-score standardization was fitted strictly on X_train and applied to transform X_test. Empirical correlation analysis revealed severe linear dependency between `energy` and `loudness` (r = 0.80). To prevent coefficient destabilization, `energy` was omitted exclusively from the OLS baseline (`taylor_baseline.csv`), whereas it was safely retained within tree architectures (`taylor_improved.csv`) which naturally tolerate collinear inputs.

---

## 📊 Model Evaluation & Performance Benchmarking

Three regression architectures were evaluated across identical partition splits (X_train vs. X_test). Hyperparameter tuning via grid search enforced a maximum tree depth boundary of 4 across tree ensembles to prevent severe training data over-memorization.

| Model Architecture | Feature Configuration | Train RMSE | Test RMSE | Train R² | Test R² | Diagnostic Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Full)** | 12 Features (Includes `energy`) | 11.4240 | 10.8603 | 0.5595 | -0.1536 | Severe Multicollinearity Damage |
| **Linear Regression (Stabilized)** | 11 Features (Excludes `energy`) | 11.4302 | 10.7815 | 0.5590 | -0.1369 | Baseline Benchmark / Negative Generalization |
| **Random Forest Regressor** | 12 Features (`max_depth=4`) | 7.6028 | 10.7317 | 0.8049 | -0.1264 | Predictive Variance Collapse |
| **Gradient Boosting Regressor** | 12 Features (`max_depth=4`) | 3.2835 | 10.7190 | 0.9636 | -0.1238 | High Overfitting / Catalog Failure |

---

## 🔍 Diagnostic Analysis: The Predictive R² Collapse

Regardless of algorithmic complexity, all modeled architectures hit a rigid predictive ceiling on unseen eras, converging to a Test RMSE around 10.7 and negative R² values (R² < 0). A negative R² mathematically indicates that the model's residual sum of squares exceeds the total sum of squares around the sample mean, performing worse than a naive benchmark tracking the mean of X_test.

### 1. Linear Coefficient Breakdown & Rigid Acoustic Rules (Figure 3)
Evaluating the learned weights of the stabilized OLS model revealed that sparse right-hand tails in asymmetric training predictors exerted extreme statistical leverage:
* **The Positive Acousticness Rule (+1.24 coefficient):** The model falsely equated high acoustic density with guaranteed commercial success. In X_test, this rule collapsed due to internal polarization: frontline acoustic hits achieved peak popularity (77–78 at acousticness 0.77–0.92), whereas deep acoustic cuts with near-absolute acousticness (0.964) collapsed to low streaming traction (popularity: 43).
* **The Negative Liveness Rule (-0.72 coefficient):** The model penalized elevated liveness scores. However, studio pop productions with arena-style reverb registered high liveness (0.32–0.38) while maintaining peak popularity (71–80), whereas actual live sessions (liveness: 0.791) yielded low platform traction (popularity: 54).

### 2. The Chronological Trap & Predictive Variance Collapse (Figure 4 & Figure 5)
Both tree ensemble models fell into an extreme chronological trap. Rather than mapping localized acoustic structures, top-level branch splits allowed `release_year` to dominate architectural utility, capturing **70.4% importance weight in Random Forest** and **69.6% in Gradient Boosting** (Figure 4).

This reliance triggered a total **Predictive Variance Collapse** (Figure 5), flattening genuine empirical standard deviations into static, era-specific predictions:
* **folklore (2020):** Actual popularity standard deviation = 11.23 -> Predicted standard deviation = 0.53 (RF) and 2.71 (GB).
* **evermore (2020):** Actual popularity standard deviation = 5.44 -> Predicted standard deviation = 0.46 (RF) and 2.60 (GB).
* **1989 (2023 Re-recording):** Actual popularity standard deviation = 5.96 -> Predicted standard deviation = 1.24 (RF) and 3.29 (GB).

---

## 💡 Key Conclusions & Future Work

Based on empirical evaluations, both research questions yielded a definitive **negative conclusion**:

* **RQ1 Conclusion (Negative):** Individual song popularity cannot be predicted using acoustic-conceptual feature matrices alone. Non-parametric complexity cannot rescue a framework when engineered features evaluate complex artistic progression through a rigid chronological lens.
* **RQ2 Conclusion (Negative):** The predictive framework cannot generalize across the broader music industry. The learned logic functions as a specialized career tracker tied strictly to Taylor Swift's unique streaming timeline and catalog replication milestones.

### 🚀 Proposed Future Directions
To overcome these structural boundaries while preserving catalog integrity, future iterations should implement two practical methodological adjustments:
1. **Distribution-Based Preprocessing:** Apply a standard log(x+1) transformation exclusively to highly skewed, non-normal acoustic parameters (e.g., speechiness, instrumentalness) to stabilize training leverage and accommodate zero-value bounds.
2. **Chronological Abstraction via External Exposure Proxies:** Replace the static, artist-specific `release_year` feature with dynamic, time-varying external metrics (e.g., media appearance frequencies, social media engagement volume, and tour scheduling intensity).

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

```bash
# 1. Clone the repository
git clone [https://github.com/Kerem-Bar/taylor-swift-spotify-sml.git](https://github.com/Kerem-Bar/taylor-swift-spotify-sml.git)

# 2. Execute Full Lifecycle & Generate Preprocessed Datasets
# (Executes cleaning, feature extraction, and saves taylor_swift_spotify.csv, taylor_baseline.csv, and taylor_improved.csv)
python Taylor_swift_smlproject.py

# 3. Run Winning Models & Reproduce Benchmark Figures
# (Trains OLS, Random Forest, Gradient Boosting and generates Figures 1-5)
python winning_models_sml2.py

## 👤 Author

**Kerem Bar**  
Master's Student in Information Sciences (Information Technology Specialization)
---
