import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

np.random.seed(42)

data = {
    "monthly_orders": [5, 8, 12, 15, 20, 25, 3, 6, 10, 18, 22, 28],
    "avg_spend": [250, 300, 450, 500, 700, 800, 180, 220, 350, 600, 750, 900],
    "avg_rating": [4.2, 4.5, 4.0, 4.6, 4.8, 4.7, 3.2, 3.5, 3.8, 4.1, 4.4, 4.6]
}

df = pd.DataFrame(data)

features = ["monthly_orders", "avg_spend", "avg_rating"]
X = df[features]

inertias = []

for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), inertias, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.show()

optimal_k = 3

model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

print("Cluster Counts:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Centroids:")
centroids = pd.DataFrame(model.cluster_centers_, columns=features)
print(centroids.round(2))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

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
plt.title("Customer Segmentation")
plt.legend()
plt.grid()
plt.show()