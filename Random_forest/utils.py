import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

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
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1,3))
    X = vectorizer.fit_transform(df["processed_text"])
    y = df["queue"]
    return X, y

def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
