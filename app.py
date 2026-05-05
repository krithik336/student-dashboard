import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Title
st.title("🎓 Student Performance Dashboard")
st.markdown("Interactive analysis of student performance")

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Clean column names
df.columns = df.columns.str.replace(" ", "_").str.lower()

# Feature engineering
df["average_score"] = (
    df["math_score"] + df["reading_score"] + df["writing_score"]
) / 3

# Sidebar filters
st.sidebar.header("Filters")

gender = st.sidebar.selectbox("Select Gender", df["gender"].unique())
prep = st.sidebar.selectbox("Test Preparation", df["test_preparation_course"].unique())

filtered_df = df[
    (df["gender"] == gender) &
    (df["test_preparation_course"] == prep)
]

# Metrics
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Average Score", round(filtered_df["average_score"].mean(), 2))
col2.metric("Highest Score", round(filtered_df["average_score"].max(), 2))
col3.metric("Lowest Score", round(filtered_df["average_score"].min(), 2))

# Charts
st.subheader("📈 Visual Insights")

col1, col2 = st.columns(2)

# Histogram
with col1:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["average_score"], kde=True, ax=ax)
    ax.set_title("Score Distribution")
    st.pyplot(fig)

# Boxplot
with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x="gender", y="average_score", data=filtered_df, ax=ax)
    ax.set_title("Gender vs Performance")
    st.pyplot(fig)

# Bar chart
st.subheader("📚 Test Preparation Impact")

fig, ax = plt.subplots()
sns.barplot(x="test_preparation_course", y="average_score", data=df, ax=ax)
ax.set_title("Effect of Test Preparation")
st.pyplot(fig)

# Heatmap
st.subheader("🔥 Correlation Between Subjects")

fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Storytelling section
st.subheader("📖 Key Findings")

st.write("""
- Students who completed test preparation scored higher  
- Reading and writing scores are strongly correlated  
- Most students fall in mid-score range  
- Gender shows slight variation in performance  
""")