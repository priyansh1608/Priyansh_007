import numpy as np
import pandas as pd
from statistics import mean, median, stdev
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

np.random.seed(42)

delivery_times = [
    25, 28, 30, 32, 35, 27, 29, 31, 34, 36,
    38, 40, 42, 45, 50, 78, 90, 105, 115, 125
]

customer_data = pd.DataFrame({
    "monthly_orders": [5, 8, 12, 15, 20, 25, 3, 6, 10, 18, 22, 28,
                       4, 7, 14, 17, 21, 24, 2, 9, 11, 16, 23, 30],
    "avg_order_value": [250, 300, 450, 500, 700, 800, 180, 220, 350, 600, 750, 900,
                        200, 280, 420, 550, 680, 850, 150, 320, 400, 580, 720, 950],
    "avg_delivery_rating": [4.2, 4.5, 4.0, 4.6, 4.8, 4.7, 3.2, 3.5, 3.8, 4.1, 4.4, 4.6,
                            3.0, 3.7, 4.0, 4.3, 4.5, 4.8, 2.8, 3.9, 4.1, 4.2, 4.5, 4.9]
})

n = 200

order_value = np.random.uniform(100, 1000, n)
delivery_distance_km = np.random.uniform(1, 20, n)
restaurant_rating = np.random.uniform(2.5, 5.0, n)

risk_score = (
    0.003 * order_value
    + 0.08 * delivery_distance_km
    - 0.8 * restaurant_rating
    + np.random.normal(0, 0.8, n)
)

cancelled = (risk_score > np.percentile(risk_score, 78)).astype(int)

classifier_data = pd.DataFrame({
    "order_value": order_value,
    "delivery_distance_km": delivery_distance_km,
    "restaurant_rating": restaurant_rating,
    "cancelled": cancelled
})

X_train = classifier_data[
    ["order_value", "delivery_distance_km", "restaurant_rating"]
]

y_train = classifier_data["cancelled"]

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, y_train)


def statistical_summary():
    order_values = [
        250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
        750, 800, 850, 900, 950, 1000, 1200, 1500
    ]

    delivery_values = delivery_times

    order_mean = mean(order_values)
    order_median = median(order_values)
    order_std = stdev(order_values)

    delivery_mean = mean(delivery_values)
    delivery_median = median(delivery_values)
    delivery_std = stdev(delivery_values)

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    print("\nOrder Value:")
    print(f"Mean: {order_mean:.2f}")
    print(f"Median: {order_median:.2f}")
    print(f"Standard Deviation: {order_std:.2f}")

    print("\nDelivery Time:")
    print(f"Mean: {delivery_mean:.2f}")
    print(f"Median: {delivery_median:.2f}")
    print(f"Standard Deviation: {delivery_std:.2f}")

    print("\nSkewness Direction:")

    if order_mean > order_median:
        print("Order Value: Right-skewed")
        print("Interpretation: A few high-value orders increase the average order value.")
    elif order_mean < order_median:
        print("Order Value: Left-skewed")
        print("Interpretation: A few low-value orders reduce the average order value.")
    else:
        print("Order Value: Approximately symmetrical")

    if delivery_mean > delivery_median:
        print("Delivery Time: Right-skewed")
        print("Interpretation: A few very long deliveries increase the average delivery time.")
    elif delivery_mean < delivery_median:
        print("Delivery Time: Left-skewed")
        print("Interpretation: A few very short deliveries reduce the average delivery time.")
    else:
        print("Delivery Time: Approximately symmetrical")


def predict_cancellation():
    print("\n" + "=" * 60)
    print("CANCELLATION RISK PREDICTION")
    print("=" * 60)

    try:
        order_value = float(input("Enter order value: "))
        delivery_distance = float(input("Enter delivery distance in km: "))
        restaurant_rating = float(input("Enter restaurant rating (1-5): "))

        if order_value <= 0:
            print("Error: Order value must be greater than 0.")
            return

        if delivery_distance <= 0:
            print("Error: Delivery distance must be greater than 0.")
            return

        if restaurant_rating < 1 or restaurant_rating > 5:
            print("Error: Restaurant rating must be between 1 and 5.")
            return

        new_order = pd.DataFrame({
            "order_value": [order_value],
            "delivery_distance_km": [delivery_distance],
            "restaurant_rating": [restaurant_rating]
        })

        prediction = classifier.predict(new_order)[0]
        probability = classifier.predict_proba(new_order)[0][1]

        print("\nPrediction Result:")

        if prediction == 1:
            print("High Cancellation Risk")
        else:
            print("Low Cancellation Risk")

        print(f"Predicted Cancellation Probability: {probability * 100:.2f}%")

    except ValueError:
        print("Error: Please enter valid numeric values.")


def customer_segmentation():
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION")
    print("=" * 60)

    features = [
        "monthly_orders",
        "avg_order_value",
        "avg_delivery_rating"
    ]

    X = customer_data[features]

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    customer_data["Cluster"] = model.fit_predict(X)

    cluster_counts = customer_data["Cluster"].value_counts().sort_index()

    print("\nNumber of Customers in Each Cluster:")

    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} customers")

    centroids = pd.DataFrame(
        model.cluster_centers_,
        columns=features
    )

    print("\nCluster Centroids:")
    print(centroids.round(2).to_string(index=True))


def main():
    while True:
        print("\n" + "=" * 60)
        print("FOOD DELIVERY CUSTOMER INTELLIGENCE SYSTEM")
        print("=" * 60)
        print("1. View Statistical Summary")
        print("2. Predict Cancellation Risk for a New Order")
        print("3. Run Customer Segmentation")
        print("4. Exit")
        print("=" * 60)

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            statistical_summary()

        elif choice == "2":
            predict_cancellation()

        elif choice == "3":
            customer_segmentation()

        elif choice == "4":
            print("\nThank you for using the Food Delivery Customer Intelligence System.")
            break

        else:
            print("\nError: Invalid choice. Please select an option from 1 to 4.")


if __name__ == "__main__":
    main()