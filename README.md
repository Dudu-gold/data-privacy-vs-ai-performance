================================================================
BALANCING DATA PRIVACY AND AI PERFORMANCE IN PREDICTIVE ANALYTICS
================================================================

DATASET
-------
UCI Heart Disease Dataset (Kaggle version)
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Save as:  heart.csv
Place in: same folder as analysis.ipynb and dashboard.py

Raw file: 1,025 rows, 14 columns.
After cleaning (723 duplicates and 1 outlier removed): 301 records
(163 disease present, 138 disease absent).

================================================================
PROJECT FILES
-------------
analysis.ipynb   - Full experiment notebook
dashboard.py     - Interactive Streamlit dashboard
requirements.txt - Python libraries needed
README.txt       - This file

================================================================
REQUIREMENTS
------------
Python 3.10 or above
pip install -r requirements.txt

================================================================
HOW TO RUN
----------
Option 1 - Jupyter Notebook:
    jupyter notebook analysis.ipynb
    Run all cells from top to bottom.

Option 2 - Streamlit Dashboard:
    streamlit run dashboard.py
    Open http://localhost:8501, upload heart.csv,
    choose epsilon and k values, click "Run Experiment".

================================================================
WHAT THE CODE DOES
------------------
Preprocessing
    Cleans messy data, one-hot encodes cp, restecg, slope, thal,
    applies z-score scaling, stratified 80/20 split
    (240 train / 61 test, random seed = 42).

Phase 1 - Baseline (no privacy)
    Linear Regression (0.5 threshold), Logistic Regression,
    Decision Tree, Random Forest.

Phase 2 - Differential Privacy
    epsilon = 0.1, 0.5, 1.0, 5.0, 10.0
    Models: Logistic Regression, Decision Tree, Random Forest
    (15 runs)
    - Logistic Regression: diffprivlib DP model
    - Decision Tree and Random Forest: Laplace noise (scale = 1/epsilon)
      added to training features (simplified approach, not a formal
      epsilon-DP guarantee)

Phase 3 - K-Anonymity
    k = 2, 5, 10
    Quasi-identifiers: age, sex, trestbps, chol, thalach
    Generalisation + suppression of groups smaller than k.
    k=10 is skipped automatically: all training records are
    suppressed, so no model can be trained.

Phase 4 - Comparative Analysis
    Heatmaps, results table, CSV export.

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score




## Results

Dataset after cleaning: 301 records (240 train / 61 test, seed 42).

### Baseline (no privacy)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Linear Regression | 0.8197 | 0.8056 | 0.8788 | 0.8406 |
| Logistic Regression | 0.8689 | 0.8788 | 0.8788 | 0.8788 |
| Decision Tree | 0.7541 | 0.7812 | 0.7576 | 0.7692 |
| Random Forest | 0.8033 | 0.8000 | 0.8485 | 0.8235 |

### Differential privacy (accuracy)

| Model | ε=0.1 | ε=0.5 | ε=1.0 | ε=5.0 | ε=10.0 |
|---|---|---|---|---|---|
| Logistic Regression | 0.5410 | 0.7869 | 0.8525 | 0.8361 | 0.8689 |
| Decision Tree | 0.4590 | 0.7377 | 0.7541 | 0.7869 | 0.7049 |
| Random Forest | 0.5410 | 0.8361 | 0.8197 | 0.8033 | 0.8197 |

### K-anonymity

| k | Records kept (of 240) | Suppressed |
|---|---|---|
| 2 | 130 | 45.8% |
| 5 | 24 | 90.0% |
| 10 | 0 | 100% (skipped) |

| Model | k=2 Accuracy | k=5 Accuracy |
|---|---|---|
| Logistic Regression | 0.8197 | 0.8197 |
| Decision Tree | 0.7049 | 0.5574 |
| Random Forest | 0.8525 | 0.7705 |

### Key findings

1. **Logistic Regression is the most privacy-resilient model.** At ε=1.0 it scored 85.25%, only 1.64 points below its baseline.
2. **ε between 1.0 and 5.0 gives the best privacy-utility balance.** At ε=0.1 every model collapsed, and the Decision Tree scored an F1 of 0.
3. **K-anonymity is limited by dataset size.** At k=5, 90% of the training data was suppressed, and at k=10 all of it was, so no model could be trained.
4. **The Decision Tree is the most sensitive model** under both techniques.
5. **Random Forest at k=2 (85.25%) beat its own baseline (80.33%).** This may be because suppression removed noisy records, but with only 61 test records it could also be chance.

### Limitations

- Only the Logistic Regression under DP uses diffprivlib's formal DP model.
  Decision Tree and Random Forest use Laplace noise added to the training
  features, which is not a formal ε-DP guarantee.
- Single small dataset (301 records), so the results may not generalise.
- Each configuration was run once, with a 61-record test set, so small
  differences between models may not be meaningful.

  
================================================================
OUTPUT FILES (saved in the same folder)
---------------------------------------
fig0_exploratory.png      - Class, age and correlation charts
fig4_1_baseline.png       - Baseline model comparison
fig4_2_3_dp_results.png   - Accuracy and F1 vs epsilon
fig4_4_5_ka_results.png   - Accuracy and F1 vs k
fig4_6_heatmap.png        - Baseline vs DP vs K-Anonymity heatmaps
all_results.csv           - All experiment results
suppression_log.csv       - K-anonymity records suppressed per k

================================================================
MESSY DATA HANDLING
-------------------
- Missing values (mode imputation)
- '?' and other placeholder values
- Duplicate rows
- Outliers (IQR method)
- Non-numeric values in numeric columns
- Different target column names across Kaggle versions
================================================================