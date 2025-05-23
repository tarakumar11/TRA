from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

stop_words = set(stopwords.words('english'))

file_path = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")


filtered_df = pd.read_csv(file_path)

filtered_df = filtered_df[filtered_df["language"] == "en"]

filtered_df.to_csv("Data/Updated_data.csv", index=False)

filtered_df = filtered_df.dropna(subset=["body"])

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

filtered_df['processed_body'] = filtered_df['body'].apply(preprocess)

dictionary = corpora.Dictionary(filtered_df['processed_body'])
corpus = [dictionary.doc2bow(text) for text in filtered_df['processed_body']]

num_topics = 10 
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

topic_to_subcategory = {
    0: "Technical Support",
    1: "Product Support",
    2: "Returns and Exchanges",
    3: "Billing and Payments",
    4: "Sales and Pre-Sales",
    5: "General Inquiry",
    6: "IT Support",
    7: "Service Outages and Maintenance",
    8: "Human Resources",
    9: "Customer Service"
}

def get_topic(document):
    tokens = preprocess(document)
    topic_distribution = lda_model.get_document_topics(dictionary.doc2bow(tokens))
    dominant_topic = max(topic_distribution, key=lambda x: x[1])
    return dominant_topic[0]

filtered_df['Topic'] = filtered_df['body'].apply(get_topic)
filtered_df['subcategory'] = filtered_df['Topic'].map(topic_to_subcategory)

category_mapping = {
    "Customer Support": ["Technical Support", "Product Support", "Returns and Exchanges", 
                         "Billing and Payments", "Sales and Pre-Sales", "General Inquiry"],
    "IT & Infrastructure": ["IT Support", "Service Outages and Maintenance"],
    "Internal Operations": ["Human Resources", "Customer Service"]
}

def get_category(subcategory):
    for category, subcategories in category_mapping.items():
        if subcategory in subcategories:
            return category
    return "Uncategorized"

filtered_df['category'] = filtered_df['subcategory'].apply(get_category)

# print(filtered_df[['body', 'category', 'subcategory']].head(20).to_string(index=False))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(filtered_df["queue"], filtered_df["subcategory"])
print(f"Accuracy: {accuracy:.2%}")
print(filtered_df[['subcategory','queue']].head(20).to_string(index=False))