import os

# File path configuration
FILE_PATH = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# Stopwords setup
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
