import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Load, Inspect and clean Data
df = pd.read_csv("Mall_Customers.csv")  # dataset from Kaggle
df = df.dropna()
print(df.head())

# Cluster customers into segments based on income and spending score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scaling the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Find Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# Apply K-Means
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
df["Cluster"] = clusters

#Visualize Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", data=df, palette="Set1", s=100)
plt.title("Customer Segments (K-Means)")
plt.show()
