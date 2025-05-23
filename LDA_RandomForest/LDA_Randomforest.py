import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load dataset
file_path = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df = pd.read_csv(file_path)

# Filter English entries and drop missing values
df = df[df["language"] == "en"].dropna(subset=["subject", "body", "queue"])

stopwords_set = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = text.replace("\n", " ").replace("\r", " ")
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords_set]

    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return " ".join(lemmatized_tokens)

# Apply preprocessing
df["processed_subject"] = df["subject"].apply(preprocess)
df["processed_body"] = df["body"].apply(preprocess)
df["processed_text"] = df["processed_subject"] + " " + df["processed_body"]

# Feature extraction with optimized parameters
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(df["processed_text"])
y = df["queue"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [15, 20, 25],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Accuracy: {accuracy:.2%}")

# # Display sample results
results = pd.DataFrame({"Queue": y_test, "Predicted": y_pred})
print(results.head(10).to_string(index=False))
