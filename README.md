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