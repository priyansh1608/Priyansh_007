import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "restaurant_rating": [4.5, 4.2, None, 3.8, 4.7, None, 4.1, 3.9, 4.8, 4.0, 3.7, 4.6],
    "payment_method": [
        "Cash", "Card", "Wallet", "Card",
        "Cash", "Wallet", "Card", "Cash",
        "Wallet", "Card", "Cash", "Wallet"
    ],
    "cuisine_type": [
        "Indian", "Chinese", "Italian", "Fast Food",
        "Indian", "Chinese", "Italian", "Fast Food",
        "Indian", "Chinese", "Italian", "Fast Food"
    ],
    "order_value": [
        250.0, 400.0, 320.0, 180.0,
        550.0, 275.0, 450.0, 200.0,
        600.0, 350.0, 225.0, 500.0
    ]
}

df = pd.DataFrame(data)

df["restaurant_rating"] = df["restaurant_rating"].fillna(
    df["restaurant_rating"].mean()
)

print("Missing values after filling:")
print(df.isnull().sum())

df = pd.get_dummies(
    df,
    columns=["payment_method", "cuisine_type"],
    dtype=int
)

scaler = MinMaxScaler()

df[["restaurant_rating", "order_value"]] = scaler.fit_transform(
    df[["restaurant_rating", "order_value"]]
)

print("\nFinal Preprocessed DataFrame:")
print(df)