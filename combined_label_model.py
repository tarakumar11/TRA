import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the preprocessed data
df = pd.read_csv("Data/preprocessed_data.csv")

# Combine 'subject' and 'body' into a single text column
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

# Create a combined label: Category + Sub-Category
df['Combined_Label'] = df['Category'] + " - " + df['Sub-Category']

# Define features and target
X = df['text']
y = df['Combined_Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

def train_and_evaluate(X_tr, y_tr, X_te, y_te, model, model_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred))
    print("\n")

# ---------- Train 3 Models on Combined Label ----------

print("### Combined Label Prediction ###\n")

# Random Forest
train_and_evaluate(X_train_vect, y_train, X_test_vect, y_test,
                   RandomForestClassifier(random_state=42), "Random Forest")

# Logistic Regression
train_and_evaluate(X_train_vect, y_train, X_test_vect, y_test,
                   LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression")

# # CatBoost
# train_and_evaluate(X_train_vect, y_train, X_test_vect, y_test,
#                    CatBoostClassifier(verbose=0, random_seed=42), "CatBoost")
