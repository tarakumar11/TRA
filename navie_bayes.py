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
from imblearn.over_sampling import SMOTE


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


file_path = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df = pd.read_csv(file_path)


df = df[df["language"] == "en"].dropna(subset=["subject", "body", "queue"])

stopwords =set(stopwords.words("english"))



stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.replace("\n", " ").replace("\r", " ")
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
    
   
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return " ".join(lemmatized_tokens)


df["processed_subject"] = df["subject"].apply(preprocess)
df["processed_body"] = df["body"].apply(preprocess)
df["processed_text"] = df["processed_subject"] + " " + df["processed_body"]

vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1,3))
X = vectorizer.fit_transform(df["processed_text"])
y = df["queue"]


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy: {accuracy:.2%}")


results = pd.DataFrame({"Queue": y_test, "Predicted": y_pred})
print(results.head(10).to_string(index=False))


