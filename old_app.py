import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Student ML Dashboard",
    page_icon="🎓",
    layout="wide"
)

# =========================
# 🌈 PREMIUM GLASS UI
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

/* Hide streamlit menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Titles */
h1 {
    color: white;
    font-size: 52px !important;
    font-weight: 700 !important;
}

h2, h3 {
    color: #00E5FF;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00C6FF, #0072FF);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    font-weight: 600;
}

/* Download Button */
.stDownloadButton > button {
    background: linear-gradient(90deg, #00C6FF, #0072FF);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.06);
    border-radius: 15px;
    padding: 10px;
}

/* Input boxes */
.stNumberInput input {
    background-color: rgba(255,255,255,0.08);
    color: white;
    border-radius: 10px;
}

/* Select box */
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 🏷 HERO SECTION
# =========================
st.markdown("""
<div style='padding:30px 10px 10px 10px;'>

<h1 style='text-align:center;'>
Smart Academic Prediction Dashboard
</h1>

<p style='text-align:center;
font-size:20px;
color:#D3D3D3;'>

Machine Learning • Analytics • Predictions • Insights

</p>

</div>
""", unsafe_allow_html=True)

# =========================
# 📂 LOAD DATASET
# =========================
df = pd.read_csv("orginaldataset.csv")

# =========================
# 🧹 CLEAN COLUMNS
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.lower()
)

# =========================
# 🩹 HANDLE MISSING VALUES
# =========================
numeric_df = df.select_dtypes(include=np.number)

df[numeric_df.columns] = numeric_df.fillna(
    numeric_df.mean()
)

# =========================
# 📄 DATASET PREVIEW
# =========================
st.subheader("📄 Dataset Preview")

if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# =========================
# 📊 DATASET OVERVIEW
# =========================
numeric_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("📄 Rows", df.shape[0])
c2.metric("📑 Columns", df.shape[1])
c3.metric("🎯 Numeric Features", len(numeric_cols))

# =========================
# 🔥 CORRELATION HEATMAP
# =========================
st.subheader("🔥 GPA Correlation Heatmap")

corr = df.corr(numeric_only=True)

if "gpa" in corr.columns:

    corr_target = corr[["gpa"]].sort_values(
        by="gpa",
        ascending=False
    )

else:

    first_col = corr.columns[0]

    corr_target = corr[[first_col]].sort_values(
        by=first_col,
        ascending=False
    )

fig, ax = plt.subplots(figsize=(5, 6))

sns.heatmap(
    corr_target,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 8},
    ax=ax
)

plt.yticks(fontsize=9)

st.pyplot(fig)

# =========================
# 🎯 TARGET VARIABLE
# =========================
st.subheader("🎯 Select Target Variable")

target = st.selectbox(
    "Choose column to predict",
    numeric_cols
)

# =========================
# 🔍 FEATURES
# =========================
features = [
    col for col in numeric_cols
    if col != target
]

X = df[features]
y = df[target]

# =========================
# ✂️ TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 🤖 MODEL TRAINING
# =========================

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeRegressor(
    random_state=42
)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(
    random_state=42
)
rf_model.fit(X_train, y_train)

# =========================
# 📈 PREDICTIONS
# =========================
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# =========================
# 📊 METRICS
# =========================
lr_score = r2_score(y_test, lr_pred)
dt_score = r2_score(y_test, dt_pred)
rf_score = r2_score(y_test, rf_pred)

# =========================
# 📋 PERFORMANCE TABLE
# =========================
st.subheader("📊 Model Performance")

results_df = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "R² Score": [
        lr_score,
        dt_score,
        rf_score
    ],

    "MAE": [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, dt_pred),
        mean_absolute_error(y_test, rf_pred)
    ],

    "RMSE": [
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, dt_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ]
})

st.dataframe(
    results_df.style.highlight_max(axis=0)
)

# =========================
# 🏆 BEST MODEL
# =========================
best_model = results_df.loc[
    results_df["R² Score"].idxmax(),
    "Model"
]

st.success(f"🏆 Best Performing Model: {best_model}")

# =========================
# 📈 VISUALIZATION SECTION
# =========================
col1, col2 = st.columns(2)

# -------------------------
# Accuracy Graph
# -------------------------
with col1:

    st.subheader("📈 Model Accuracy")

    fig, ax = plt.subplots(figsize=(4, 3))

    models = [
        "Linear",
        "Decision Tree",
        "Random Forest"
    ]

    scores = [
        lr_score,
        dt_score,
        rf_score
    ]

    ax.bar(models, scores)

    ax.set_ylabel("R² Score")
    ax.set_facecolor("#1c1c1c")

    st.pyplot(fig)

# -------------------------
# Actual vs Predicted
# -------------------------
with col2:

    st.subheader("📉 Actual vs Predicted")

    selected_model_graph = st.selectbox(
        "Select Model",
        [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    if selected_model_graph == "Linear Regression":
        preds = lr_pred

    elif selected_model_graph == "Decision Tree":
        preds = dt_pred

    else:
        preds = rf_pred

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(y_test, preds)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_facecolor("#1c1c1c")

    st.pyplot(fig)

# =========================
# 🔮 PREDICTION SECTION
# =========================
st.subheader("🔮 Predict Values")

selected_model = st.selectbox(
    "Choose Prediction Model",
    [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ]
)

input_data = []

cols = st.columns(2)

for i, feature in enumerate(features):

    value = cols[i % 2].number_input(
        f"{feature}",
        value=float(df[feature].mean())
    )

    input_data.append(value)

# =========================
# 🎯 PREDICTION BUTTON
# =========================
if st.button("Predict"):

    input_array = np.array(
        input_data
    ).reshape(1, -1)

    if selected_model == "Linear Regression":

        prediction = lr_model.predict(
            input_array
        )

        accuracy = lr_score

    elif selected_model == "Decision Tree":

        prediction = dt_model.predict(
            input_array
        )

        accuracy = dt_score

    else:

        prediction = rf_model.predict(
            input_array
        )

        accuracy = rf_score

    st.success(
        f"🎯 Predicted {target}: "
        f"{round(prediction[0], 2)}"
    )

    st.info(
        f"📊 Model Accuracy: "
        f"{round(accuracy, 3)}"
    )

# =========================
# 📥 DOWNLOAD REPORT
# =========================
csv = results_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Model Report",
    data=csv,
    file_name="model_results.csv",
    mime="text/csv"
)

# =========================
# 🚀 PROJECT SUMMARY
# =========================
st.markdown("""
<div style='
background: rgba(255,255,255,0.08);
padding:20px;
border-radius:15px;
margin-top:20px;
text-align:center;'>

<h3 style='color:#00E5FF;'>
🚀 Project Summary
</h3>

<p>
This dashboard compares multiple Machine Learning algorithms
for predicting student academic performance using real-world data.
</p>

</div>
""", unsafe_allow_html=True)

# =========================
# 🚀 FOOTER
# =========================
# =========================
# ⚠️ DISCLAIMER
# =========================
st.markdown("""
<div style='
background: rgba(255,255,255,0.06);
padding:15px;
border-radius:12px;
margin-top:20px;
font-size:14px;
color:#D3D3D3;
'>

⚠️ <b>Disclaimer:</b>  
This dashboard provides predictive insights using machine learning algorithms.  
Prediction accuracy depends on dataset quality, selected features, and model behavior.  
Results are intended for analytical purposes only and may not fully represent real-world outcomes.

</div>
""", unsafe_allow_html=True)