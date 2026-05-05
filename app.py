import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Student Dashboard", layout="wide")

# Title
st.title("🎓 Student Performance Dashboard")
st.markdown("Interactive analysis + prediction of student performance")

# Load data
df = pd.read_csv("StudentsPerformance.csv")

# Clean column names
df.columns = df.columns.str.replace(" ", "_").str.lower()

# Feature engineering
df["average_score"] = (
    df["math_score"] + df["reading_score"] + df["writing_score"]
) / 3

# =========================
# 🤖 Train Model
# =========================
X = df[["reading_score", "writing_score"]]
y = df["math_score"]

model = LinearRegression()
model.fit(X, y)

# =========================
# 🔍 Sidebar Filters
# =========================
st.sidebar.header("Filters")

gender = st.sidebar.selectbox("Gender", df["gender"].unique())
prep = st.sidebar.selectbox("Preparation", df["test_preparation_course"].unique())

filtered_df = df[
    (df["gender"] == gender) &
    (df["test_preparation_course"] == prep)
]

# =========================
# 📊 Metrics
# =========================
st.subheader("📊 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Average Score", round(filtered_df["average_score"].mean(), 2))
c2.metric("Highest Score", round(filtered_df["average_score"].max(), 2))
c3.metric("Lowest Score", round(filtered_df["average_score"].min(), 2))

# =========================
# 📈 Charts
# =========================
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

# =========================
# 📄 Data Preview
# =========================
st.subheader("📄 Dataset Preview")

if st.checkbox("Show Raw Data"):
    st.write(df.head())

# =========================
# ⬇ Download Button
# =========================
st.download_button(
    label="Download Dataset",
    data=df.to_csv(index=False),
    file_name="students_data.csv",
    mime="text/csv"
)

# =========================
# 📖 Insights
# =========================
st.subheader("📖 Key Insights")

st.success("Students who completed test preparation scored higher.")
st.info("Reading and writing scores are strongly correlated.")
st.warning("Most students fall in mid-performance range.")

# =========================
# 🔮 Prediction Section
# =========================
st.subheader("🔮 Predict Math Score")

st.markdown("Enter reading and writing scores:")

col1, col2 = st.columns(2)

reading_input = col1.number_input("Reading Score", 0, 100, 50)
writing_input = col2.number_input("Writing Score", 0, 100, 50)

if st.button("Predict"):
    prediction = model.predict([[reading_input, writing_input]])
    st.success(f"Predicted Math Score: {round(prediction[0], 2)}")

# Footer
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Student Data Analytics Project")