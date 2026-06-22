"""
Spam Message Intelligence — Classifier Comparison App
Converted from PR-4.ipynb into an interactive Streamlit web app.

Models covered: KNN, SVM, Naive Bayes
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Message Intelligence",
    page_icon="📩",
    layout="wide"
)

st.title("📩 Message Intelligence — Spam Classifier Comparison")
st.caption("KNN vs SVM vs Naive Bayes — interactive version of PR-4.ipynb")

# ----------------------------------------------------------------------
# Sidebar — data loading & settings
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Message_Intelligence_Dataset_5200.csv (or similar)", type=["csv"]
)

target_col_default = "spam_label"
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20, step=5) / 100
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🔧 Model Parameters")
knn_k = st.sidebar.slider("KNN: number of neighbors (K)", 1, 25, 5)
knn_metric = st.sidebar.selectbox("KNN: distance metric", ["euclidean", "manhattan", "minkowski"])
svm_kernel = st.sidebar.selectbox("SVM: kernel", ["linear", "rbf", "poly"])

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.info("👈 Upload the dataset CSV from the sidebar to get started. "
            "Showing instructions below in the meantime.")
    df = None

if df is not None:
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

    target_col = st.selectbox(
        "Select target column",
        options=df.columns,
        index=list(df.columns).index(target_col_default) if target_col_default in df.columns else 0
    )

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    feature_cols = st.multiselect(
        "Select feature columns (numeric)",
        options=numeric_cols,
        default=numeric_cols
    )

    run_button = st.button("🚀 Run Models", type="primary")

    if run_button and feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # Preprocessing
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

        st.subheader("📊 Train / Test Split")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train samples", X_train.shape[0])
        c2.metric("Test samples", X_test.shape[0])
        c3.metric("Features", X_train.shape[1])
        c4.metric("Test size", f"{int(test_size*100)}%")

        # ----------------------------------------------------------------
        # Train models
        # ----------------------------------------------------------------
        knn = KNeighborsClassifier(n_neighbors=knn_k, metric=knn_metric)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        svm = SVC(kernel=svm_kernel)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)

        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)

        models = {"KNN": y_pred_knn, "SVM": y_pred_svm, "Naive Bayes": y_pred_nb}

        # ----------------------------------------------------------------
        # Metrics table
        # ----------------------------------------------------------------
        results = pd.DataFrame({
            "Model": list(models.keys()),
            "Accuracy": [accuracy_score(y_test, p) for p in models.values()],
            "Precision": [precision_score(y_test, p, zero_division=0) for p in models.values()],
            "Recall": [recall_score(y_test, p, zero_division=0) for p in models.values()],
            "F1 Score": [f1_score(y_test, p, zero_division=0) for p in models.values()],
        })

        st.subheader("🏆 Model Comparison")
        st.dataframe(
            results.style.highlight_max(
                axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#b7f7c1"
            ),
            use_container_width=True
        )

        best_model = results.loc[results["F1 Score"].idxmax(), "Model"]
        st.success(f"✅ Best performing model by F1 Score: **{best_model}**")

        # ----------------------------------------------------------------
        # Bar chart comparison
        # ----------------------------------------------------------------
        st.subheader("📈 Performance Chart")
        fig, ax = plt.subplots(figsize=(9, 5))
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        x = np.arange(len(results["Model"]))
        width = 0.2
        for i, metric in enumerate(metric_names):
            ax.bar(x + (i - 1.5) * width, results[metric], width, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(results["Model"])
        ax.set_ylabel("Score")
        ax.set_title("KNN vs SVM vs Naive Bayes")
        ax.legend()
        st.pyplot(fig)

        # ----------------------------------------------------------------
        # Confusion matrices
        # ----------------------------------------------------------------
        st.subheader("🧮 Confusion Matrices")
        cols = st.columns(3)
        for col, (name, pred) in zip(cols, models.items()):
            cm = confusion_matrix(y_test, pred)
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
            im = ax_cm.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
            ax_cm.set_title(name)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            col.pyplot(fig_cm)

        # ----------------------------------------------------------------
        # Misclassified samples
        # ----------------------------------------------------------------
        st.subheader("❌ Misclassified Samples (KNN)")
        mis_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred_knn
        })
        mis_cases = mis_df[mis_df["Actual"] != mis_df["Predicted"]]
        st.write(f"Number of misclassified messages: **{len(mis_cases)}**")
        st.dataframe(mis_cases.head(10), use_container_width=True)

        # ----------------------------------------------------------------
        # KNN: effect of K
        # ----------------------------------------------------------------
        st.subheader("🔁 Effect of K on KNN Accuracy")
        k_values = list(range(1, 21))
        k_acc = []
        for k in k_values:
            knn_k_model = KNeighborsClassifier(n_neighbors=k, metric=knn_metric)
            knn_k_model.fit(X_train, y_train)
            k_acc.append(accuracy_score(y_test, knn_k_model.predict(X_test)))
        fig_k, ax_k = plt.subplots(figsize=(8, 4))
        ax_k.plot(k_values, k_acc, marker="o")
        ax_k.set_xlabel("K (number of neighbors)")
        ax_k.set_ylabel("Accuracy")
        ax_k.set_title("KNN Accuracy vs K")
        st.pyplot(fig_k)

        best_k_idx = int(np.argmax(k_acc))
        st.info(f"Best K = **{k_values[best_k_idx]}** with accuracy **{k_acc[best_k_idx]:.4f}**")

    elif run_button and not feature_cols:
        st.warning("Please select at least one feature column.")

else:
    st.markdown("""
    ### Expected dataset columns
    The original notebook used `Message_Intelligence_Dataset_5200.csv` with features such as:

    - message_length, word_count, num_urls, num_digits, num_special_chars
    - spam_keyword_score, legit_keyword_score
    - sender_activity_score, sender_account_age_days, messages_sent_last_24h
    - hour_of_day, day_of_week
    - **spam_label** (target: 0 = not spam, 1 = spam)

    Upload your CSV from the sidebar to begin training and comparing KNN, SVM, and Naive Bayes models.
    """)

st.markdown("---")
st.caption("Built with Streamlit • Converted from PR-4.ipynb")
