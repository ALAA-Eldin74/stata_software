import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------------- Page Config ----------------
st.set_page_config(page_title="Pro ML Platform", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("Pro ML Platform")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload a dataset from the sidebar to get started")
    st.stop()

# ---------------- Load Data ----------------
if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["Data Overview", "Visualization & ML"])

# ================= TAB 1 =================
with tab1:
    st.subheader("Dataset Overview & Statistics")

    # ---- Explanation ----
    st.info("""
    📊 **Statistical Description**
    
    - Numerical Statistics: Mean, Std, Min, Max, Quartiles  
    - Categorical Statistics: Unique values, Most frequent value  
    - Helps understand data distribution before Visualization & ML
    """)

    # ---- Buttons ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_preview = st.button("Show Data Preview")
    with col2:
        show_info = st.button("Show Dataset Info")
    with col3:
        show_num_stats = st.button("Numerical Statistics")
    with col4:
        show_cat_stats = st.button("Categorical Statistics")

    # ---- Data Preview ----
    if show_preview:
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)

    # ---- Dataset Info ----
    if show_info:
        st.subheader("Dataset Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.dataframe(info_df, use_container_width=True)

    # ---- Numerical Statistics ----
    if show_num_stats:
        st.subheader("Numerical Statistical Summary")
        num_desc = df.select_dtypes(include=np.number).describe().T
        st.dataframe(num_desc, use_container_width=True)

    # ---- Categorical Statistics ----
    if show_cat_stats:
        st.subheader("Categorical Statistical Summary")
        cat_cols = df.select_dtypes(exclude=np.number).columns

        cat_summary = []
        for col in cat_cols:
            cat_summary.append({
                "Column": col,
                "Count": df[col].count(),
                "Unique": df[col].nunique(),
                "Most Frequent": df[col].mode()[0] if not df[col].mode().empty else None,
                "Frequency": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else None
            })

        cat_df = pd.DataFrame(cat_summary)
        st.dataframe(cat_df, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("Visualization")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    x_col = st.selectbox("X Axis", df.columns)
    y_col = st.selectbox("Y Axis", num_cols if num_cols else df.columns)
    chart = st.selectbox("Chart Type", ["Scatter", "Bar", "Box", "Histogram", "Pie"])

    if chart == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col)
    elif chart == "Bar":
        fig = px.bar(df, x=x_col, y=y_col)
    elif chart == "Box":
        fig = px.box(df, x=x_col, y=y_col)
    elif chart == "Histogram":
        fig = px.histogram(df, x=x_col)
    else:
        fig = px.pie(df, names=x_col)

    if st.button("Generate Chart"):
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- ML ----------------
    st.markdown("---")
    st.subheader("Machine Learning")

    target = st.selectbox("Target Column", df.columns)
    features = st.multiselect(
        "Features",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )

    X = df[features].copy()
    y = df[target]

    # ---- Detect Problem Type ----
    if y.dtype == object or y.nunique() <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"

    st.info(f"Detected Problem Type: **{problem_type.upper()}**")

    # ---- Encode target ----
    if problem_type == "classification" and y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    # ---- Handle Missing Values ----
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            X[col].fillna(X[col].mean(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0], inplace=True)

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if problem_type == "classification" else None
    )

    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # ---- Models ----
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier()
        }
        metric = st.selectbox("Metric", ["accuracy", "f1", "recall", "precision"])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200)
        }
        metric = "r2"
        cv = 3

    # ---- Train ----
    if st.button("Train & Compare Models"):
        results = []
        best_score = -999
        best_model = None
        best_name = ""

        for name, model in models.items():
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=metric)
            mean_score = scores.mean()

            results.append({"Model": name, "Score": mean_score})

            if mean_score > best_score:
                best_score = mean_score
                best_model = pipe
                best_name = name

        res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.subheader("Model Comparison")
        st.dataframe(res_df, use_container_width=True)

        st.success(f"Best Model: **{best_name}**")

        # ---- Evaluation ----
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        if problem_type == "classification":
            st.metric("Test Accuracy", f"{accuracy_score(y_test, preds):.2%}")
            st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True))
            st.text(classification_report(y_test, preds))
        else:
            st.metric("R2 Score", f"{r2_score(y_test, preds):.3f}")
            st.metric("RMSE", f"{mean_squared_error(y_test, preds, squared=False):.3f}")

        # ---- Feature Importance ----
        if "Random Forest" in best_name:
            model = best_model.named_steps["model"]

            cat_feature_names = []
            if cat_features:
                cat_feature_names = list(
                    best_model.named_steps["prep"]
                    .named_transformers_["cat"]
                    .get_feature_names_out(cat_features)
                )

            feature_names = num_features + cat_feature_names

            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            st.subheader("Feature Importance")
            st.plotly_chart(
                px.bar(
                    imp_df.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h"
                ),
                use_container_width=True
            )

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Pro ML Platform | Streamlit")