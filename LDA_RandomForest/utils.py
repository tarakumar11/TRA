import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text, stopwords):
    text = text.replace("\n", " ").replace("\r", " ")
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords]

    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    return " ".join(lemmatized_tokens)

def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    X = vectorizer.fit_transform(df["processed_text"])
    y = df["queue"]
    return X, y

def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def perform_grid_search(X_train, y_train):
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [15, 20, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_
