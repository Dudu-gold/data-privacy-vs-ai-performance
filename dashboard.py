"""
Streamlit Dashboard — Balancing Data Privacy and AI Performance
Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privacy vs AI Performance",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Balancing Data Privacy & AI Performance")
st.markdown("**UCI Heart Disease Dataset** — Differential Privacy & K-Anonymity Analysis")
st.markdown("---")

SEED = 42

# ── Helpers ────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    return {
        "Label":     label,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

def clean_dataset(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Detect target
    target_col = None
    for c in ["target", "condition", "num", "heart_disease"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]

    df[target_col] = (df[target_col] > 0).astype(int)
    df.replace(["?", "NA", "na", "N/A", "", " "], np.nan, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    df.drop_duplicates(inplace=True)

    # IQR outlier removal
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]

    return df, target_col

# ── Sidebar — Upload ────────────────────────────────────────────────────
st.sidebar.header("📁 Dataset")
uploaded = st.sidebar.file_uploader("Upload heart.csv", type=["csv"])

if uploaded is None:
    st.info("👈 Upload your **heart.csv** file in the sidebar to begin.")
    st.markdown("""
    **Download the dataset from Kaggle:**
    https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

    Save it as `heart.csv` and upload it above.
    """)
    st.stop()

# ── Load & clean ────────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded)
df, target_col = clean_dataset(df_raw.copy())

st.sidebar.success(f"✅ Loaded: {df.shape[0]} records, {df.shape[1]} columns")
st.sidebar.markdown(f"**Target column:** `{target_col}`")

# ── Sidebar — Settings ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
epsilon_values = st.sidebar.multiselect(
    "Epsilon values (DP)",
    [0.1, 0.5, 1.0, 5.0, 10.0],
    default=[0.1, 0.5, 1.0, 5.0, 10.0]
)
k_values = st.sidebar.multiselect(
    "K values (K-Anonymity)",
    [2, 5, 10],
    default=[2, 5, 10]
)

run_btn = st.sidebar.button("▶ Run Experiment", type="primary", use_container_width=True)

# ── Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview",
    "🔵 Phase 1 — Baseline",
    "🔐 Phase 2 — Differential Privacy",
    "👥 Phase 3 — K-Anonymity",
    "📈 Phase 4 — Comparison"
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1: Dataset Overview
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Disease Present", int(df[target_col].sum()))
    col4.metric("Disease Absent", int((df[target_col] == 0).sum()))

    st.markdown("**First 5 rows:**")
    st.dataframe(df.head(), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Class Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        df[target_col].value_counts().plot(kind="bar", ax=ax,
            color=["#1C2B4A", "#888888"], edgecolor="white")
        ax.set_xticklabels(["No Disease", "Disease"], rotation=0)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Feature Correlation with Target**")
        fig, ax = plt.subplots(figsize=(5, 4))
        corr = df.corr()[target_col].drop(target_col).sort_values()
        corr.plot(kind="barh", ax=ax,
            color=["#C0392B" if v < 0 else "#1C2B4A" for v in corr])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Dataset Statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# Preprocessing (shared across tabs)
# ─────────────────────────────────────────────────────────────────────────
X = df.drop(columns=[target_col])
y = df[target_col]

cat_cols = [c for c in ["cp", "restecg", "slope", "thal", "ca"] if c in X.columns]
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Tune DT depth
best_depth, best_score = 3, 0
for depth in range(2, 12):
    s = cross_val_score(
        DecisionTreeClassifier(max_depth=depth, random_state=SEED),
        X_train_s, y_train, cv=5, scoring="f1"
    ).mean()
    if s > best_score:
        best_score, best_depth = s, depth

if not run_btn:
    with tab2:
        st.info("Click **▶ Run Experiment** in the sidebar to start.")
    with tab3:
        st.info("Click **▶ Run Experiment** in the sidebar to start.")
    with tab4:
        st.info("Click **▶ Run Experiment** in the sidebar to start.")
    with tab5:
        st.info("Click **▶ Run Experiment** in the sidebar to start.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: Baseline
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Phase 1 — Baseline Model Training (No Privacy)")

    with st.spinner("Training baseline models..."):
        baseline_results = []

        # Linear Regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_s, y_train)
        y_pred_lr = (lin_reg.predict(X_test_s) >= 0.5).astype(int)
        baseline_results.append({
            "Label": "Linear Regression",
            "Accuracy":  round(accuracy_score(y_test, y_pred_lr), 4),
            "Precision": round(precision_score(y_test, y_pred_lr, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred_lr, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred_lr, zero_division=0), 4),
        })

        log_reg = LogisticRegression(max_iter=1000, random_state=SEED)
        log_reg.fit(X_train_s, y_train)
        baseline_results.append(evaluate(log_reg, X_test_s, y_test, "Logistic Regression"))

        dt = DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)
        dt.fit(X_train_s, y_train)
        baseline_results.append(evaluate(dt, X_test_s, y_test, "Decision Tree"))

        rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
        rf.fit(X_train_s, y_train)
        baseline_results.append(evaluate(rf, X_test_s, y_test, "Random Forest"))

        baseline_df = pd.DataFrame(baseline_results)

    st.dataframe(baseline_df.set_index("Label"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(baseline_df))
    w = 0.18
    colors = ["#1C2B4A", "#555555", "#999999", "#CCCCCC"]
    for i, (m, c) in enumerate(zip(["Accuracy","Precision","Recall","F1-Score"], colors)):
        ax.bar(x + i*w, baseline_df[m], w, label=m, color=c, edgecolor="white")
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(baseline_df["Label"], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.set_title("Baseline Model Performance Comparison", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────
# TAB 3: Differential Privacy
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Phase 2 — Differential Privacy")

    if not epsilon_values:
        st.warning("Please select at least one epsilon value in the sidebar.")
        st.stop()

    dp_results = []
    with st.spinner("Applying differential privacy..."):
        data_norm = float(np.max(np.linalg.norm(X_train_s, axis=1)))
        for eps in sorted(epsilon_values):
            noise = 1.0 / eps
            X_noisy = X_train_s + np.random.laplace(0, noise, X_train_s.shape)

            try:
                from diffprivlib.models import LogisticRegression as DPLR
                dp_lr = DPLR(epsilon=eps, data_norm=data_norm, max_iter=1000)
                dp_lr.fit(X_train_s, y_train)
                dp_results.append(evaluate(dp_lr, X_test_s, y_test, f"Logistic Regression (ε={eps})"))
            except Exception:
                m = LogisticRegression(max_iter=1000, random_state=SEED)
                m.fit(X_noisy, y_train)
                dp_results.append(evaluate(m, X_test_s, y_test, f"Logistic Regression (ε={eps})"))

            m = DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)
            m.fit(X_noisy, y_train)
            dp_results.append(evaluate(m, X_test_s, y_test, f"Decision Tree (ε={eps})"))

            m = RandomForestClassifier(n_estimators=100, random_state=SEED)
            m.fit(X_noisy, y_train)
            dp_results.append(evaluate(m, X_test_s, y_test, f"Random Forest (ε={eps})"))

    dp_df = pd.DataFrame(dp_results)
    st.dataframe(dp_df.set_index("Label"), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    eps_sorted = sorted(epsilon_values)
    for prefix, ls, color in [
        ("Logistic Regression", "-o", "#1C2B4A"),
        ("Decision Tree",       "-s", "#555555"),
        ("Random Forest",       "-^", "#999999"),
    ]:
        accs = [dp_df[dp_df["Label"] == f"{prefix} (ε={e})"]["Accuracy"].values[0]
                for e in eps_sorted if f"{prefix} (ε={e})" in dp_df["Label"].values]
        f1s  = [dp_df[dp_df["Label"] == f"{prefix} (ε={e})"]["F1-Score"].values[0]
                for e in eps_sorted if f"{prefix} (ε={e})" in dp_df["Label"].values]
        eps_plot = [e for e in eps_sorted if f"{prefix} (ε={e})" in dp_df["Label"].values]
        axes[0].plot(eps_plot, accs, ls, label=prefix, color=color, linewidth=2, markersize=7)
        axes[1].plot(eps_plot, f1s,  ls, label=prefix, color=color, linewidth=2, markersize=7)

    for ax, title in zip(axes, ["Accuracy vs Epsilon", "F1-Score vs Epsilon"]):
        ax.set_xscale("log")
        ax.set_xlabel("Epsilon (ε)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────
# TAB 4: K-Anonymity
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Phase 3 — K-Anonymity")

    if not k_values:
        st.warning("Please select at least one k value in the sidebar.")
        st.stop()

    ka_results = []
    suppression_log = []
    qi_cols = [c for c in ["age", "sex", "trestbps", "chol", "thalach"] if c in df.columns]

    with st.spinner("Applying k-anonymity..."):
        for k in sorted(k_values):
            train_df = X_train.copy()
            train_df[target_col] = y_train.values

            if "age"      in train_df.columns: train_df["age"]      = (train_df["age"]      // 10) * 10
            if "trestbps" in train_df.columns: train_df["trestbps"] = (train_df["trestbps"] // 20) * 20
            if "chol"     in train_df.columns: train_df["chol"]     = (train_df["chol"]     // 50) * 50
            if "thalach"  in train_df.columns: train_df["thalach"]  = (train_df["thalach"]  // 20) * 20

            existing_qi = [c for c in qi_cols if c in train_df.columns]
            if existing_qi:
                counts = train_df.groupby(existing_qi)[target_col].transform("count")
                before = len(train_df)
                train_df = train_df[counts >= k]
                after = len(train_df)
                suppression_log.append({"k": k, "Before": before, "After": after,
                                         "Suppressed": before-after,
                                         "Suppressed %": round(100*(before-after)/before, 1)})

            if len(train_df) < 20:
                st.warning(f"k={k}: Too few records after suppression. Skipped.")
                continue

            Xk = train_df.drop(columns=[target_col])
            yk = train_df[target_col]
            if cat_cols:
                existing_cat = [c for c in cat_cols if c in Xk.columns]
                if existing_cat:
                    Xk = pd.get_dummies(Xk, columns=existing_cat, drop_first=False)
            Xk = Xk.reindex(columns=X_train.columns, fill_value=0)
            sk = StandardScaler()
            Xk_s = sk.fit_transform(Xk)
            Xt_s = sk.transform(X_test)

            for name, model in [
                ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
                ("Decision Tree",       DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)),
                ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=SEED)),
            ]:
                model.fit(Xk_s, yk)
                ka_results.append(evaluate(model, Xt_s, y_test, f"{name} (k={k})"))

    ka_df = pd.DataFrame(ka_results)

    if suppression_log:
        st.markdown("**Suppression Summary:**")
        st.dataframe(pd.DataFrame(suppression_log), use_container_width=True)

    if len(ka_df) > 0:
        st.dataframe(ka_df.set_index("Label"), use_container_width=True)

        available_k = sorted(ka_df["Label"].str.extract(r"k=(\d+)")[0].dropna().astype(int).unique())
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for prefix, ls, color in [
            ("Logistic Regression", "-o", "#1C2B4A"),
            ("Decision Tree",       "-s", "#555555"),
            ("Random Forest",       "-^", "#999999"),
        ]:
            accs = [ka_df[ka_df["Label"] == f"{prefix} (k={k})"]["Accuracy"].values[0]
                    for k in available_k if f"{prefix} (k={k})" in ka_df["Label"].values]
            f1s  = [ka_df[ka_df["Label"] == f"{prefix} (k={k})"]["F1-Score"].values[0]
                    for k in available_k if f"{prefix} (k={k})" in ka_df["Label"].values]
            k_plot = [k for k in available_k if f"{prefix} (k={k})" in ka_df["Label"].values]
            axes[0].plot(k_plot, accs, ls, label=prefix, color=color, linewidth=2, markersize=7)
            axes[1].plot(k_plot, f1s,  ls, label=prefix, color=color, linewidth=2, markersize=7)
        for ax, title in zip(axes, ["Accuracy vs K", "F1-Score vs K"]):
            ax.set_xlabel("K Value")
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No K-Anonymity results generated. Try reducing k values.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 5: Comparison
# ─────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Phase 4 — Comparative Analysis")

    summary_rows = []
    for _, row in baseline_df.iterrows():
        summary_rows.append({"Model": row["Label"], "Technique": "Baseline",
                             "Accuracy": row["Accuracy"], "F1-Score": row["F1-Score"]})

    for prefix, model_name in [
        ("Logistic Regression", "Logistic Regression"),
        ("Decision Tree",       "Decision Tree"),
        ("Random Forest",       "Random Forest"),
    ]:
        mid_eps = sorted(epsilon_values)[len(epsilon_values)//2]
        row = dp_df[dp_df["Label"] == f"{prefix} (ε={mid_eps})"]
        if len(row) > 0:
            summary_rows.append({"Model": model_name, "Technique": f"DP (ε={mid_eps})",
                                 "Accuracy": row["Accuracy"].values[0],
                                 "F1-Score": row["F1-Score"].values[0]})

    if len(ka_df) > 0:
        available_k2 = sorted(ka_df["Label"].str.extract(r"k=(\d+)")[0].dropna().astype(int).unique())
        mid_k = available_k2[len(available_k2)//2] if available_k2 else None
        if mid_k:
            for prefix, model_name in [
                ("Logistic Regression", "Logistic Regression"),
                ("Decision Tree",       "Decision Tree"),
                ("Random Forest",       "Random Forest"),
            ]:
                row = ka_df[ka_df["Label"] == f"{prefix} (k={mid_k})"]
                if len(row) > 0:
                    summary_rows.append({"Model": model_name, "Technique": f"KA (k={mid_k})",
                                         "Accuracy": row["Accuracy"].values[0],
                                         "F1-Score": row["F1-Score"].values[0]})

    summary_df = pd.DataFrame(summary_rows)

    col1, col2 = st.columns(2)
    for col, metric in zip([col1, col2], ["Accuracy", "F1-Score"]):
        with col:
            try:
                pivot = summary_df.pivot(index="Model", columns="Technique", values=metric)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Greys",
                            linewidths=0.5, ax=ax, vmin=0.4, vmax=1.0)
                ax.set_title(f"{metric} Heatmap", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Could not generate {metric} heatmap: {e}")

    st.markdown("**Full Comparison Table:**")
    st.dataframe(summary_df, use_container_width=True)

    # Download button
    all_df = pd.concat([baseline_df, dp_df, ka_df], ignore_index=True)
    csv = all_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Full Results CSV",
        data=csv,
        file_name="all_results.csv",
        mime="text/csv"
    )
