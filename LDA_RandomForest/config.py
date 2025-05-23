import os
import nltk
from nltk.corpus import stopwords

# File path configuration
FILE_PATH = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# Stopwords setup
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))