import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

np.random.seed(42)

n = 180

monthly_orders = np.random.randint(1, 31, n)
avg_order_value = np.random.uniform(100, 1000, n)
avg_delivery_rating = np.random.uniform(1.0, 5.0, n)

df = pd.DataFrame({
    "monthly_orders": monthly_orders,
    "avg_order_value": avg_order_value,
    "avg_delivery_rating": avg_delivery_rating
})

features = [
    "monthly_orders",
    "avg_order_value",
    "avg_delivery_rating"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = range(2, 10)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(list(k_values))
plt.grid()
plt.show()

optimal_k = 3

final_model = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_model.fit_predict(X_scaled)

print("=" * 50)
print("CUSTOMER CLUSTER COUNTS")
print("=" * 50)
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Centroids:")
centroids = scaler.inverse_transform(final_model.cluster_centers_)

centroid_df = pd.DataFrame(
    centroids,
    columns=features
)

centroid_df.index = [
    f"Cluster {i}"
    for i in range(optimal_k)
]

print(centroid_df.round(2))

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

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