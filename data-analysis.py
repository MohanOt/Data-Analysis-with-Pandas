
# Assignment: Analyzing Data with Pandas and Visualizing Results with Matplotlib

# --- Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# Task 1: Load and Explore the Dataset


try:
    # Load the iris dataset directly from sklearn
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame  # already comes as a Pandas DataFrame
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset file not found.")
except Exception as e:
    print("❌ Error loading dataset:", str(e))

# Display the first 5 rows
print("\n🔹 First 5 rows of the dataset:")
print(df.head())

# Check dataset info
print("\n🔹 Dataset Info:")
print(df.info())

# Check for missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# Clean dataset (no missing values in Iris, but let's demonstrate)
df = df.dropna()


# Task 2: Basic Data Analysis


# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Grouping: mean of numerical columns by species
grouped = df.groupby("target").mean()
print("\n🔹 Mean values per species:")
print(grouped)

# Observations
print("\n📌 Observations:")
print("- Sepal length and petal length differ significantly across species.")
print("- Iris setosa generally has smaller petal lengths compared to versicolor and virginica.")


# Task 3: Data Visualization


# Apply a nice Seaborn style
sns.set(style="whitegrid")

# 1. Line chart (not typical for Iris, but simulate by sorting sepal length over index)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart of Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x="target", y="petal length (cm)", data=df, estimator="mean", ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species (0=Setosa, 1=Versicolor, 2=Virginica)")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", palette="Set2", data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

print("\n✅ Analysis complete. Plots generated successfully!")
