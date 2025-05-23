import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import config
import utils

# Load data
df = pd.read_csv(config.FILE_PATH)

# Filter English language and remove missing values
df = df[df["language"] == "en"].dropna(subset=["subject", "body", "queue"])

# Preprocess text
df["processed_subject"] = df["subject"].apply(lambda text: utils.preprocess(text, config.STOPWORDS))
df["processed_body"] = df["body"].apply(lambda text: utils.preprocess(text, config.STOPWORDS))
df["processed_text"] = df["processed_subject"] + " " + df["processed_body"]

# Vectorization
X, y = utils.vectorize_text(df)

# Balance dataset
X_resampled, y_resampled = utils.balance_classes(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning & Model Selection
best_model = utils.perform_grid_search(X_train, y_train)

# Model evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Accuracy: {accuracy:.2%}")

# Display results
results = pd.DataFrame({"Queue": y_test, "Predicted": y_pred})
print(results.head(10).to_string(index=False))
