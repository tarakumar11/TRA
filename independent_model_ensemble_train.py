import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

# Individual models for Category prediction
rf_cat = RandomForestClassifier(random_state=42)
lr_cat = LogisticRegression(max_iter=1000, random_state=42)
# cb_cat = CatBoostClassifier(verbose=0, random_seed=42)

# Individual models for Sub-Category prediction
rf_sub = RandomForestClassifier(random_state=42)
lr_sub = LogisticRegression(max_iter=1000, random_state=42)
# cb_sub = CatBoostClassifier(verbose=0, random_seed=42)

print("### Category Prediction ###\n")
train_and_evaluate(X_train_vect, y_cat_train, X_test_vect, y_cat_test, rf_cat, "Random Forest", "Category")
train_and_evaluate(X_train_vect, y_cat_train, X_test_vect, y_cat_test, lr_cat, "Logistic Regression", "Category")
# train_and_evaluate(X_train_vect, y_cat_train, X_test_vect, y_cat_test, cb_cat, "CatBoost", "Category")

# Ensemble for Category prediction
ensemble_cat = VotingClassifier(
    estimators=[('rf', rf_cat),  ('lr', lr_cat)], #('cb', cb_cat)],
    voting='soft'  # or 'hard'
)
train_and_evaluate(X_train_vect, y_cat_train, X_test_vect, y_cat_test, ensemble_cat, "Ensemble", "Category")

print("### Sub-Category Prediction ###\n")
train_and_evaluate(X_train_vect, y_sub_train, X_test_vect, y_sub_test, rf_sub, "Random Forest", "Sub-Category")
train_and_evaluate(X_train_vect, y_sub_train, X_test_vect, y_sub_test, lr_sub, "Logistic Regression", "Sub-Category")
# train_and_evaluate(X_train_vect, y_sub_train, X_test_vect, y_sub_test, cb_sub, "CatBoost", "Sub-Category")

# Ensemble for Sub-Category prediction
ensemble_sub = VotingClassifier(
    estimators=[('rf', rf_sub), ('lr', lr_sub)], # ('cb', cb_sub)],
    voting='soft'  # or 'hard'
)
train_and_evaluate(X_train_vect, y_sub_train, X_test_vect, y_sub_test, ensemble_sub, "Ensemble", "Sub-Category")
