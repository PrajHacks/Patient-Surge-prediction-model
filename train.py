import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("er_full_output_dataset_1000.csv")

X = df[[
    "temperature",
    "aqi",
    "festival",
    "current_patients",
    "humidity",
    "day_type"
]]

y = df["er_load_category"]   # Target: 0 = Low, 1 = Moderate, 2 = High, 3 = Critical


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


with open("er_load_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as er_load_model.pkl")
