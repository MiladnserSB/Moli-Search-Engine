# Preprocessing and Tokenization Helpers for Indexing Service
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# Download NLTK data if not already present
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Contraction dictionary from the supervisor's guidelines/report
contractions_dict = {
    "u": "you",
    "r": "are",
    "wanna": "want to",
    "canna": "cannot",
    "don't": "do not",
    "didn't": "did not",
    "it's": "it is",
    "i'm": "i am",
}

contractions_re = re.compile('(%s)' % '|'.join(map(re.escape, contractions_dict.keys())))

def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

# Text cleaning function from the supervisor's guidelines/report
def processing(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = expand_contractions(text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b\d{1,2}\b", "", text)
    text = re.sub(r"\b\d{5,}\b", "", text)
    return text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

# Tokenizer/POS-lemmatizer from the supervisor's guidelines/report
def tokenize(text):
    tokens = word_tokenize(text)
    tokens_pos = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tokens_pos
        if word not in stop_words and len(word) > 1
    ]
    return lemmatized

# Lightweight cleaning for BERT embeddings
def clean_light(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
