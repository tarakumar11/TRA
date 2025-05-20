import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


nltk.download("stopwords")
nltk.download("punkt")


# Load dataset
file_path = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df = pd.read_csv(file_path)

# Filter for English emails and remove missing values
df = df[df["language"] == "en"].dropna(subset=["subject", "body"])

# Define stop words
stop_words = set(stopwords.words("english"))
# print(stop_words)
# stop_words = ['Dear Customer Support', 'Team']




def preprocess(text):
    text = text.replace("\n", " ").replace("\r", " ")
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

df["processed_subject"] = df["subject"].apply(preprocess)
df["processed_body"] = df["body"].apply(preprocess)

# Combine subject and body for better topic detection
df["processed_text"] = df["processed_subject"] + " " + df["processed_body"]

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer( ngram_range=(2, 2))
# vectorizer = CountVectorizer( ngram_range=(2, 2))
X = vectorizer.fit_transform(df["processed_text"])
y = df["queue"] 


X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

results = pd.DataFrame({"class":y_test, "predicted":y_pred})
print(results.head(10).to_string(index=False))
