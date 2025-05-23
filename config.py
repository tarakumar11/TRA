import os
import nltk
from nltk.corpus import stopwords


FILE_PATH = os.path.abspath("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")


nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
