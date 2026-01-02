import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/sample_data.csv")

df["thickness_margin"] = (
    df["remaining_thickness_mm"] - df["min_thickness_required_mm"]
)

df["pressure_ratio"] = (
    df["operating_pressure_bar"] / df["design_pressure_bar"]
)

X = df.drop(columns=["failure"])
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully")
