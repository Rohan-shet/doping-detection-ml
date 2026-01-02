import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# === STEP 1: Load your dataset ===

df = pd.read_csv("enriched_athlete_dataset.csv")


 # replace if needed

# === STEP 2: Select features and label ===
features = ['vo2_max', 'hemoglobin', 'testosterone']
label = 'is_doped'

# Remove rows with missing data
df.dropna(subset=features + [label], inplace=True)

X = df[features]
y = df[label]

# === STEP 3: Split into training and test sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 4: Train a logistic regression model ===
model = LogisticRegression()
model.fit(X_train, y_train)

# === STEP 5: Evaluate the model ===
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === STEP 6: Predict on full dataset ===
df['is_predicted_doped'] = model.predict(X)

# === STEP 7: Save output with predictions ===
df.to_csv("ml_doping_detection_output.csv", index=False)
print("✅ Predictions saved to ml_doping_detection_output.csv")
