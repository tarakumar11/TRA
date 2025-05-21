import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your preprocessed data
df = pd.read_csv("Data/preprocessed_data.csv")

# Combine 'subject' and 'body' into a single text feature
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

# Define features and targets
X = df['text']
y_category = df['Category']
y_subcategory = df['Sub-Category']

# Split data (80% train, 20% test), stratify on Category for balanced splits
X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = train_test_split(
    X, y_category, y_subcategory, test_size=0.2, random_state=42, stratify=y_category
)

# Vectorize text with TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

def train_and_evaluate(X_tr, y_tr, X_te, y_te, model, model_name, task_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"--- {model_name} - {task_name} ---")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred))
    print("\n")

# ---------- Model 1: Predict Category ----------
print("### Category Prediction ###\n")

# Random Forest
train_and_evaluate(X_train_vect, y_cat_train, X_test_vect, y_cat_test,
                   RandomForestClassifier(random_state=42), "Random Forest", "Category")

# Logistic Regression
train_and_evaluate(X_train_vect, y_cat_train, X_test_vect, y_cat_test,
                   LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression", "Category")

# # CatBoost (set verbose=0 for no output)
# train_and_evaluate(X_train_vect, y_cat_train, X_test_vect, y_cat_test,
#                    CatBoostClassifier(verbose=0, random_seed=42), "CatBoost", "Category")


# ---------- Model 2: Predict Sub-Category ----------
print("### Sub-Category Prediction ###\n")

# Random Forest
train_and_evaluate(X_train_vect, y_sub_train, X_test_vect, y_sub_test,
                   RandomForestClassifier(random_state=42), "Random Forest", "Sub-Category")

# Logistic Regression
train_and_evaluate(X_train_vect, y_sub_train, X_test_vect, y_sub_test,
                   LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression", "Sub-Category")

# # CatBoost
# train_and_evaluate(X_train_vect, y_sub_train, X_test_vect, y_sub_test,
#                    CatBoostClassifier(verbose=0, random_seed=42), "CatBoost", "Sub-Category")
