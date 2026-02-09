import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensuring necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    """Handles cleaning and tokenization of raw text."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text):
        # Lowercase and remove punctuation/numbers
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r'\d+', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        cleaned_tokens = [w for w in tokens if w not in self.stop_words]

        return " ".join(cleaned_tokens)