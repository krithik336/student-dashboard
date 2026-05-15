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
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Academic Analytics Dashboard",
    layout="wide"
)

# =========================
# THEME TOGGLE
# =========================
theme = st.toggle(
    "Dark Mode",
    value=False
)

# =========================
# DYNAMIC COLORS
# =========================
if theme:

    bg_color = "#0F172A"
    card_color = "#1E293B"
    text_color = "#F8FAFC"
    secondary_text = "#CBD5E1"
    border_color = "#334155"

    metric_text = "#FFFFFF"
    nav_text = "#FFFFFF"

else:

    bg_color = "#F4F7FE"
    card_color = "#FFFFFF"
    text_color = "#111827"
    secondary_text = "#6B7280"
    border_color = "#E5E7EB"

    metric_text = "#111827"
    nav_text = "#111827"

# =========================
# MODERN UI
# =========================
st.markdown(f"""
<style>

/* Main App */
.stApp {{
    background: {bg_color};
    color: {text_color};
}}

/* Global Text */
body {{
    color: {text_color};
}}

/* Streamlit Markdown */
[data-testid="stMarkdownContainer"] * {{
    color: {text_color} !important;
}}

/* Hide Branding */
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}
header {{visibility:hidden;}}

/* Remove Sidebar */
section[data-testid="stSidebar"] {{
    display:none;
}}

/* Headers */
h1, h2, h3, h4 {{
    color: {text_color} !important;
}}

/* Labels */
label {{
    color: {text_color} !important;
    font-weight: 600;
}}

/* Top Header */
.top-header {{
    background: {card_color};
    padding: 25px;
    border-radius: 24px;
    border: 1px solid {border_color};
    margin-bottom: 20px;
}}

/* Dashboard Title */
.dashboard-title {{
    font-size: 42px;
    font-weight: 700;
    color: {text_color};
}}

/* Subtitle */
.dashboard-subtitle {{
    color: {secondary_text};
    font-size: 16px;
}}

/* Metric Cards */
div[data-testid="metric-container"] {{
    background: {card_color};
    border-radius: 18px;
    padding: 18px;
    border: 1px solid {border_color};
}}

/* Metric Values */
div[data-testid="metric-container"] > div {{
    color: {metric_text} !important;
    font-weight: 700;
}}

/* Metric Labels */
div[data-testid="metric-container"] label {{
    color: {secondary_text} !important;
}}

/* Buttons */
.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px;
    font-weight: 600;
    transition: 0.3s ease;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, #1D4ED8, #1E40AF);
    transform: translateY(-2px);
}}

/* Download Button */
.stDownloadButton > button {{
    width: 100%;
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 14px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    box-shadow: 0px 4px 12px rgba(37,99,235,0.25);
    transition: 0.3s ease;
}}

.stDownloadButton > button:hover {{
    background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
    transform: translateY(-2px);
}}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] {{
    background: {card_color};
    color: {text_color} !important;
    border-radius: 12px;
    border: 1px solid {border_color};
}}

/* Dropdown Text */
div[data-baseweb="popover"] * {{
    color: black !important;
}}

/* Number Input */
.stNumberInput input {{
    background: {card_color};
    color: {text_color} !important;
    border-radius: 12px;
    border: 1px solid {border_color};
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    background: {card_color};
    border-radius: 18px;
    border: 1px solid {border_color};
}}

/* Dataframe Text */
[data-testid="stDataFrame"] * {{
    color: {text_color} !important;
}}

/* Radio Buttons */
.stRadio label {{
    color: {nav_text} !important;
    font-weight: 600;
    font-size: 16px;
}}

/* Radio Group */
div[role="radiogroup"] {{
    display:flex;
    gap:20px;
}}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("orginaldataset.csv")

# =========================
# CLEAN COLUMNS
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.lower()
)

# =========================
# HANDLE MISSING VALUES
# =========================
numeric_df = df.select_dtypes(include=np.number)

df[numeric_df.columns] = numeric_df.fillna(
    numeric_df.mean()
)

# =========================
# TOP DASHBOARD
# =========================
st.markdown(f"""
<div class="top-header">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
flex-wrap:wrap;
">

<div>

<div class="dashboard-title">
Academic Analytics Dashboard
</div>

<div class="dashboard-subtitle">
Machine Learning • EDA • Predictions • Insights
</div>

</div>

<div style="
display:flex;
gap:10px;
align-items:center;
">


</div>

</div>

</div>
""", unsafe_allow_html=True)

# =========================
# KPI DASHBOARD
# =========================
numeric_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Rows", df.shape[0])
k2.metric("Columns", df.shape[1])
k3.metric("Features", len(numeric_cols))
k4.metric("ML Models", 3)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# TOP NAVIGATION
# =========================
section = st.radio(
    "Navigation",
    [
        "Home",
        "EDA Analysis",
        "Machine Learning",
        "Predictions"
    ],
    horizontal=True
)

st.markdown("---")

# =========================
# HOME
# =========================
if section == "Home":

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        annot=False,
        ax=ax
    )

    st.pyplot(fig)

# =========================
# EDA ANALYSIS
# =========================
if section == "EDA Analysis":

    st.header("Exploratory Data Analysis")

    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    if "gpa" in df.columns:

        st.subheader("GPA Distribution")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(
            df["gpa"],
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

# =========================
# MACHINE LEARNING
# =========================
if section == "Machine Learning":

    st.header("Machine Learning Models")

    target = st.selectbox(
        "Select Target Variable",
        numeric_cols
    )

    features = [
        col for col in numeric_cols
        if col != target
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    lr_model = LinearRegression()
    dt_model = DecisionTreeRegressor(random_state=42)
    rf_model = RandomForestRegressor(random_state=42)

    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    results_df = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "R² Score": [
            r2_score(y_test, lr_pred),
            r2_score(y_test, dt_pred),
            r2_score(y_test, rf_pred)
        ],

        "MAE": [
            mean_absolute_error(y_test, lr_pred),
            mean_absolute_error(y_test, dt_pred),
            mean_absolute_error(y_test, rf_pred)
        ]
    })

    st.subheader("Model Performance")
    st.dataframe(results_df)

# =========================
# PREDICTIONS
# =========================
if section == "Predictions":

    st.header("Smart Predictions")

    target = st.selectbox(
        "Select Target",
        numeric_cols
    )

    features = [
        col for col in numeric_cols
        if col != target
    ]

    X = df[features]
    y = df[target]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    st.subheader("Enter Feature Values")

    input_data = []

    cols = st.columns(2)

    for i, feature in enumerate(features):

        value = cols[i % 2].number_input(
            feature,
            value=float(df[feature].mean())
        )

        input_data.append(value)

    if st.button("Predict"):

        prediction = model.predict(
            np.array(input_data).reshape(1, -1)
        )

        st.success(
            f"Predicted {target}: "
            f"{round(prediction[0], 2)}"
        )

# =========================
# DOWNLOAD REPORT
# =========================
st.download_button(
    label="Download Report",
    data=df.to_csv(index=False),
    file_name="model_results.csv",
    mime="text/csv"
)

# =========================
# DISCLAIMER
# =========================
st.markdown(f"""
<div style='
background:{card_color};
padding:18px;
border-radius:18px;
margin-top:20px;
border:1px solid {border_color};
color:{secondary_text};
'>

<b>Disclaimer:</b><br><br>

This dashboard provides predictive insights using machine learning algorithms.
Prediction accuracy depends on dataset quality and selected features.
Results are intended for analytical and educational purposes only.

</div>
""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown(f"""
<div style='
text-align:center;
color:{secondary_text};
margin-top:20px;
padding:10px;
'>

Built using Streamlit & Scikit-learn<br>
Developed by <b>Krithik</b>

</div>
""", unsafe_allow_html=True)