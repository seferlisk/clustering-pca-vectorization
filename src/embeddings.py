import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText


class EmbeddingEngine:
    """
    Handles the transformation of text into various vector formats.
    """

    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.models = {}

    def get_tfidf_embeddings(self, corpus):
        """Standard TF-IDF Vectorization."""
        print("Generating TF-IDF embeddings...")
        vectorizer = TfidfVectorizer(max_features=5000)
        return vectorizer.fit_transform(corpus).toarray()

    def _get_gensim_average_vector(self, model, corpus):
        """
        Helper to convert a list of documents into a single vector
        by averaging the word vectors.
        """
        vectors = []
        for doc in corpus:
            # Tokenize the document (assuming it's a space-separated string)
            words = doc.split()
            word_vectors = [model.wv[w] for w in words if w in model.wv]

            if len(word_vectors) > 0:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                # If no words are in the vocabulary, return a vector of zeros
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)

    def get_word2vec_embeddings(self, corpus):
        """Trains a Word2Vec model and generates document vectors."""
        print("Generating Word2Vec embeddings...")
        tokenized_corpus = [doc.split() for doc in corpus]
        model = Word2Vec(sentences=tokenized_corpus, vector_size=self.vector_size,
                         window=5, min_count=1, workers=4)
        return self._get_gensim_average_vector(model, corpus)

    def get_fasttext_embeddings(self, corpus):
        """Trains a FastText model and generates document vectors."""
        print("Generating FastText embeddings...")
        tokenized_corpus = [doc.split() for doc in corpus]
        model = FastText(sentences=tokenized_corpus, vector_size=self.vector_size,
                         window=5, min_count=1, workers=4)
        return self._get_gensim_average_vector(model, corpus)