import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

data = {
    "monthly_orders": [5, 8, 12, 15, 20, 25, 3, 6, 10, 18, 22, 28],
    "avg_spend": [250, 300, 450, 500, 700, 800, 180, 220, 350, 600, 750, 900],
    "avg_rating": [4.2, 4.5, 4.0, 4.6, 4.8, 4.7, 3.2, 3.5, 3.8, 4.1, 4.4, 4.6]
}

df = pd.DataFrame(data)

features = ["monthly_orders", "avg_spend", "avg_rating"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_values = range(2, 10)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(list(k_values))
plt.grid()
plt.show()

optimal_k = 3

model = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)

print("=" * 50)
print("CLUSTER COUNTS")
print("=" * 50)
print(df["Cluster"].value_counts().sort_index())

centroids = scaler.inverse_transform(model.cluster_centers_)

centroid_df = pd.DataFrame(
    centroids,
    columns=features
)

print("\n" + "=" * 50)
print("CLUSTER CENTROIDS")
print("=" * 50)
print(centroid_df.round(2))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

for cluster in range(optimal_k):
    points = X_pca[df["Cluster"] == cluster]
    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Cluster {cluster}"
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segmentation using K-Means and PCA")
plt.legend()
plt.grid()
plt.show()