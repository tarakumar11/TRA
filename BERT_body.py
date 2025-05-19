# BERT embeddings and KMeans for topic classification for body
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import nltk
nltk.download('stopwords')
file_path = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

filtered_df = pd.read_csv(file_path)

filtered_df = filtered_df[filtered_df["language"] == "en"]

filtered_df.to_csv("Data/Updated_data.csv", index=False)

filtered_df = filtered_df.dropna(subset=["body"])

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)  # Convert token list back into a cleaned sentence
   

filtered_df['processed_body'] = filtered_df['body'].apply(preprocess)

# Initialize the BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate BERT embeddings
sentence_embeddings = model.encode(filtered_df['processed_body'].tolist())

# KMeans for topic classification for clustering 
num_topics = 10  
kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
filtered_df['Topic'] = kmeans.fit_predict(sentence_embeddings)

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