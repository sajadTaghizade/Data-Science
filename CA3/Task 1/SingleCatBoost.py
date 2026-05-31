import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

train = pd.read_excel("./Dataset/train.xlsx")
test = pd.read_excel("./Dataset/student_test.xlsx")

def clean_text(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()

def clean_binary(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in ["yes", "true", "y", "1"]:
        return 1
    if x in ["no", "false", "n", "0"]:
        return 0
    return np.nan

def to_numeric_safe(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

numeric_cols = [
    "transaction_amount",
    "avg_transaction_amount_7d",
    "transaction_frequency_24h",
    "failed_transaction_count_24h",
    "account_age_days"
]

train = to_numeric_safe(train, numeric_cols)
test = to_numeric_safe(test, numeric_cols)

binary_cols = [
    "is_international",
    "unusual_amount_flag",
    "multiple_transactions_short_time",
    "high_risk_device_flag"
]

for col in binary_cols:
    train[col] = train[col].apply(clean_binary)
    test[col] = test[col].apply(clean_binary)

for df in [train, test]:
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")

    df["hour"] = df["transaction_timestamp"].dt.hour
    df["day"] = df["transaction_timestamp"].dt.day
    df["month"] = df["transaction_timestamp"].dt.month
    df["dayofweek"] = df["transaction_timestamp"].dt.dayofweek

    df.drop("transaction_timestamp", axis=1, inplace=True)

cat_cols = ["payment_method", "device_type", "location", "merchant_category"]

for df in [train, test]:
    for col in cat_cols:
        df[col] = df[col].apply(clean_text).fillna("missing")

for df in [train, test]:

    df["amount_to_avg_ratio"] = df["transaction_amount"] / (df["avg_transaction_amount_7d"] + 1e-6)

    df["amount_diff"] = df["transaction_amount"] - df["avg_transaction_amount_7d"]

    df["risk_score"] = (
        df["is_international"].fillna(0) * 2 +
        df["unusual_amount_flag"].fillna(0) * 2 +
        df["multiple_transactions_short_time"].fillna(0) * 1.5 +
        df["high_risk_device_flag"].fillna(0) * 2
    )

    df["missing_values_count"] = df.isna().sum(axis=1)


X = train.drop(["label", "transaction_id"], axis=1)
y = train["label"]

X_test = test.drop(["id", "transaction_id"], axis=1)

categorical_cols = [
    "customer_id",
    "payment_method",
    "device_type",
    "location",
    "merchant_category"
]

for col in categorical_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_features = [X.columns.get_loc(col) for col in categorical_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=42,
    class_weights=[1, 6],
    verbose=200
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)

val_probs = model.predict_proba(X_val)[:, 1]

best_f1 = 0
best_t = 0.5

for t in np.arange(0.1, 0.9, 0.01):
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nBest threshold:", best_t)
print("Best F1:", best_f1)

final_val_preds = (val_probs >= best_t).astype(int)

print("\nConfusion Matrix:\n", confusion_matrix(y_val, final_val_preds))
print("\nReport:\n", classification_report(y_val, final_val_preds))

model.fit(X, y, cat_features=cat_features, verbose=200)

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_t).astype(int)

submission = pd.DataFrame({
    "id": test["id"],
    "label": test_preds
})

submission.to_csv("submission.csv", index=False)

print("\n✅ submission.csv created successfully")