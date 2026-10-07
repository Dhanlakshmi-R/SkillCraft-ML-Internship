# SkillCraft Technology — Machine Learning Internship

## Task 2: Customer Segmentation using K-Means Clustering

### 📌 Project Overview

This project is part of my **Machine Learning Internship at SkillCraft Technology**.

The objective of this task is to perform **customer segmentation using K-Means Clustering**, an unsupervised machine learning algorithm. The customers are grouped into different segments based on their **Annual Income** and **Spending Score**.

Customer segmentation helps businesses understand customer behavior and develop targeted marketing strategies for different groups.

---

## 🎯 Objective

The main objectives of this project are:

* Understand customer purchasing behavior.
* Explore the customer dataset using exploratory data analysis.
* Identify meaningful customer segments.
* Apply the **K-Means Clustering** algorithm.
* Determine the optimal number of clusters using the **Elbow Method**.
* Visualize the resulting customer segments.
* Interpret the clusters from a business perspective.

---

## 📊 Dataset

The project uses the **Mall Customers Dataset**.

The dataset contains information about customers, including:

| Feature                | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| CustomerID             | Unique identification number of the customer                  |
| Gender                 | Gender of the customer                                        |
| Age                    | Age of the customer                                           |
| Annual Income (k$)     | Annual income of the customer in thousands of dollars         |
| Spending Score (1-100) | Spending score assigned based on customer purchasing behavior |

### Features Used for Clustering

For this project, the following two features were selected:

* **Annual Income (k$)**
* **Spending Score (1-100)**

These features were selected because they provide useful information about a customer's financial capacity and spending behavior.

---

## 🧠 Machine Learning Algorithm

### K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** that divides data points into a predefined number of clusters.

The algorithm works by:

1. Selecting the number of clusters `K`.
2. Initializing cluster centroids.
3. Assigning each data point to the nearest centroid.
4. Recalculating the centroid of each cluster.
5. Repeating the process until the clusters stabilize.

In this project, K-Means is used to group customers with similar income and spending behavior.

---

## 🔍 Project Workflow

The project follows these steps:

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Exploratory Data Analysis
   ↓
Elbow Method
   ↓
Selecting Optimal K
   ↓
K-Means Clustering
   ↓
Cluster Visualization
   ↓
Cluster Analysis
   ↓
Business Interpretation
```

---

## 🛠️ Technologies and Libraries Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Development Environment

* Google Colab / Jupyter Notebook
* GitHub

---

## ⚙️ Implementation Steps

### 1. Import Required Libraries

The required Python libraries are imported for data manipulation, visualization, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
```

---

### 2. Load the Dataset

The Mall Customers dataset is loaded using Pandas.

```python
df = pd.read_csv("Mall_Customers.csv")

df.head()
```

---

### 3. Explore the Dataset

The dataset was examined using:

```python
df.shape
df.info()
df.describe()
```

This helped understand:

* Number of records
* Number of features
* Data types
* Statistical characteristics

---

### 4. Check Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

This ensures that the selected features are suitable for clustering.

---

### 5. Check Duplicate Records

Duplicate records were checked using:

```python
df.duplicated().sum()
```

---

### 6. Select Features

The following features were selected:

```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

These features were used as the input to the K-Means algorithm.

---

### 7. Exploratory Data Analysis

Visualizations were created to understand the distribution of:

* Annual Income
* Spending Score

A scatter plot was also created to observe the relationship between the two selected features.

```python
plt.figure(figsize=(8,6))

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)']
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Distribution")

plt.show()
```

---

## 📐 Finding the Optimal Number of Clusters

### Elbow Method

The **Elbow Method** was used to determine a suitable value of `K`.

WCSS (Within-Cluster Sum of Squares) was calculated for different numbers of clusters.

```python
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

The WCSS values were then visualized:

```python
plt.figure(figsize=(8,6))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()
```

The point where the reduction in WCSS begins to slow down indicates a suitable number of clusters.

For the standard Mall Customers dataset, **5 clusters** are commonly used for customer segmentation.

---

## 🤖 Applying K-Means Clustering

The K-Means model was created using the selected number of clusters.

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X)
```

The resulting cluster labels were added to the dataset:

```python
df['Cluster'] = kmeans.labels_
```

---

## 📊 Cluster Visualization

The customer segments were visualized using a scatter plot.

```python
plt.figure(figsize=(10,7))

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=df['Cluster'],
    s=60
)

centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=200,
    marker='X'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")

plt.show()
```

The visualization shows the different customer groups identified by the K-Means algorithm.

---

## 📈 Results

The K-Means algorithm successfully divided the customers into **5 segments** based on their annual income and spending score.

The identified clusters can be analyzed based on their average income and spending behavior.

The cluster summary can be generated using:

```python
cluster_summary = df.groupby('Cluster')[
    ['Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print(cluster_summary)
```

### Result Summary

The clusters generally represent different combinations of income and spending behavior, such as:

* Low Income — Low Spending
* Low Income — High Spending
* Medium Income — Medium Spending
* High Income — Low Spending
* High Income — High Spending

> **Note:** Cluster numbers are assigned by the K-Means algorithm and do not have a fixed meaning. The interpretation should be based on the actual cluster averages obtained from the model.

---

## 💼 Business Interpretation

Customer segmentation can help businesses create targeted strategies for different customer groups.

### 1. High Income — High Spending

These customers have both high purchasing capacity and high spending activity.

**Possible strategy:**

* Premium products
* Exclusive offers
* Loyalty programs
* Personalized recommendations

---

### 2. High Income — Low Spending

These customers have a high income but relatively low spending activity.

**Possible strategy:**

* Personalized promotions
* Product recommendations
* Special discounts
* Engagement campaigns

The goal is to encourage these customers to increase their spending.

---

### 3. Low Income — High Spending

These customers have relatively lower income but demonstrate strong spending behavior.

**Possible strategy:**

* Affordable product recommendations
* Discounts
* Value-for-money offers
* Loyalty rewards

---

### 4. Low Income — Low Spending

These customers show relatively low income and low spending activity.

**Possible strategy:**

* Budget-friendly products
* Discounts
* Promotional campaigns
* Entry-level product offers

---

### 5. Medium Income — Medium Spending

These customers show moderate income and spending behavior.

**Possible strategy:**

* Regular promotions
* Cross-selling
* Personalized product suggestions
* Loyalty programs

---

## 📁 Project Structure

```text
SkillCraft-ML-Internship/
│
├── Task-1-House-Price-Prediction/
│
└── Task-2-Customer-Segmentation/
    │
    ├── Mall_Customers.csv
    ├── SCT_ML_Task2.ipynb
    ├── customer_segments.csv
    ├── README.md
    │
    └── images/
        ├── income_distribution.png
        ├── spending_distribution.png
        ├── elbow_method.png
        └── customer_clusters.png
```

---

## 📌 Key Learnings

Through this project, I learned:

* The difference between supervised and unsupervised learning.
* How K-Means clustering works.
* How to select relevant features for clustering.
* How to use the Elbow Method to determine the number of clusters.
* How to visualize customer segments.
* How machine learning can be applied to customer behavior analysis.
* How customer segmentation can support business decision-making.

---

## 🚀 Future Improvements

The project can be further improved by:

* Using additional customer features such as age and gender.
* Comparing different clustering algorithms.
* Applying feature scaling where appropriate.
* Using silhouette score to evaluate cluster quality.
* Developing an interactive customer segmentation dashboard.
* Using the segments for personalized marketing recommendations.

---

## 👩‍💻 Internship Task

**Program:** SkillCraft Technology — Machine Learning Internship
**Task:** Task 2 — Customer Segmentation
**Algorithm:** K-Means Clustering
**Type:** Unsupervised Machine Learning

---

## ⭐ Conclusion

This project demonstrates how **K-Means Clustering** can be used to identify meaningful customer segments based on income and spending behavior.

The resulting segments provide useful insights that can help businesses understand their customers and design more targeted marketing strategies.
