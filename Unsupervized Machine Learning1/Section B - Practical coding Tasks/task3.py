import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

n = 150

order_value = np.random.uniform(100, 1000, n)
delivery_distance_km = np.random.uniform(1, 20, n)
hour_of_day = np.random.randint(0, 24, n)
restaurant_rating = np.random.uniform(2.5, 5.0, n)

cancelled = np.zeros(n, dtype=int)

cancelled_indices = np.random.choice(
    n,
    size=33,
    replace=False
)

cancelled[cancelled_indices] = 1

df = pd.DataFrame({
    "order_value": order_value,
    "delivery_distance_km": delivery_distance_km,
    "hour_of_day": hour_of_day,
    "restaurant_rating": restaurant_rating,
    "cancelled": cancelled
})

X = df[
    [
        "order_value",
        "delivery_distance_km",
        "hour_of_day",
        "restaurant_rating"
    ]
]

y = df["cancelled"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("                 Predicted")
print("                 0       1")
print(f"Actual 0       {tn:5d}   {fp:5d}")
print(f"Actual 1       {fn:5d}   {tp:5d}")

print("=" * 60)
print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)