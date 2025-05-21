import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from catboost import CatBoostClassifier  # ← CatBoost import is commented

from sklearn.metrics import accuracy_score, classification_report

# ---------------- Configuration ---------------- #
# Choose model type: "rf" for Random Forest, "lr" for Logistic Regression, "cat" for CatBoost
MODEL_TYPE = "rf"   # change to "rf" or "cat" as needed

def get_model():
    if MODEL_TYPE == "rf":
        return RandomForestClassifier(random_state=42)
    elif MODEL_TYPE == "lr":
        return LogisticRegression(max_iter=1000, random_state=42)
    # elif MODEL_TYPE == "cat":
    #     return CatBoostClassifier(verbose=0, random_seed=42)
    else:
        raise ValueError("Invalid model type. Use 'rf', 'lr', or 'cat'.")

# ---------------- Load and Prepare Data ---------------- #
df = pd.read_csv("Data/preprocessed_data.csv")
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

X = df['text']
y_cat = df['Category']
y_sub = df['Sub-Category']

X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = train_test_split(
    X, y_cat, y_sub, test_size=0.2, random_state=42, stratify=y_cat
)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# ---------------- Step 1: Train Category Model ---------------- #
cat_model = get_model()
cat_model.fit(X_train_vect, y_cat_train)
predicted_categories = cat_model.predict(X_test_vect)

print(f"\n### [Step 1] Category Prediction ({MODEL_TYPE.upper()}) ###")
print("Accuracy:", accuracy_score(y_cat_test, predicted_categories))
print(classification_report(y_cat_test, predicted_categories))

# ---------------- Step 2: Train Sub-Category Models Per Category ---------------- #
subcategory_models = {}
subcat_vectorizers = {}

df_train = pd.DataFrame({
    'text': X_train,
    'Category': y_cat_train,
    'Sub-Category': y_sub_train
})

for category in df_train['Category'].unique():
    subset = df_train[df_train['Category'] == category]
    X_sub = subset['text']
    y_subcat = subset['Sub-Category']

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_sub_vect = vec.fit_transform(X_sub)

    sub_model = get_model()
    sub_model.fit(X_sub_vect, y_subcat)

    subcategory_models[category] = sub_model
    subcat_vectorizers[category] = vec

# ---------------- Predict Sub-Category Based on Predicted Category ---------------- #
final_sub_preds = []

df_test = pd.DataFrame({
    'text': X_test,
    'true_category': y_cat_test,
    'true_subcategory': y_sub_test,
    'predicted_category': predicted_categories
})

for _, row in df_test.iterrows():
    predicted_cat = row['predicted_category']
    text = row['text']

    if predicted_cat in subcategory_models:
        vec = subcat_vectorizers[predicted_cat]
        model = subcategory_models[predicted_cat]
        X_vec = vec.transform([text])
        pred_sub = model.predict(X_vec)[0]
    else:
        pred_sub = "Unknown"

    final_sub_preds.append(pred_sub)

# ---------------- Evaluation ---------------- #
print(f"\n### [Step 2] Sub-Category Prediction ({MODEL_TYPE.upper()}) ###")
print("Accuracy:", accuracy_score(df_test['true_subcategory'], final_sub_preds))
print(classification_report(df_test['true_subcategory'], final_sub_preds))
