from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

import re

import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords as nltkstopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, addstopwords=None, simplify="lemma"):
        self.addstopwords = addstopwords
        self.simplify = simplify
        self.stop_words = set(nltkstopwords.words('english'))

        if addstopwords is not None:
            for w in addstopwords:
                self.stop_words.append(w)

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        """
        Preprocesses a single text string.
        Can select to use either PorterStemmer or WordNetLemmatizer by setting root to "stem" or "lemma"
        Can add new stopwords to the existing list of stop words by adding a list of words to addstopwords
        Preprocessing steps:
        -Tokenize with Regexp tokenizer
        -Remove stopwords
        -Turn all words into root words
        """
        # setup
        new_text = str(text)

        new_text = re.sub(r'\d', ' ', new_text)
        tokenizer = RegexpTokenizer(r"\w+")
        stop_words = set(nltkstopwords.words('english'))

        # text preprocessing
        tokens = tokenizer.tokenize(new_text)
        if self.simplify == "lemma":
            tokens = [self.lemmatizer.lemmatize(t, pos="v") for t in tokens]
        else:
            tokens = [self.stemmer.stem(t) for t in tokens]
            # tokens = [word for word in tokens if word.isalpha()] # Just keep the words and removes numbers (only uncomment if planning to use)
        tokens = [w.lower() for w in tokens]  # words to lower case
        tokens = [w for w in tokens if not w in stop_words]  # Removing the stop words

        new_text = ' '.join(tokens)

        return new_text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Do transformations and return
        if type(X) == list:
            new_X = pd.Series(X)
        elif type(X) == str:
            new_X = pd.Series([X])
        else:
            new_X = X.copy()

        return new_X.apply(self.preprocess_text)

if __name__ == '__main__':

    #read the process data frame with the title and the abstract of the patent
    df = pd.read_csv('ABST_final_wTTL.zip')

    # include a new column with the text from title and abstract
    df['all_text'] = df['TTL'] + ' ' + df['PAL']



    pipe = Pipeline(
        [('preprocess', TextPreprocessor()), ('tf_idf', TfidfVectorizer()), ('svd', TruncatedSVD(n_components=128))])

    X_pal = pipe.fit_transform(df['all_text'])

    df_emb = pd.DataFrame(X_pal)
    df_emb.index = df['PID']
    df_emb.to_csv('tfidf_svd_all.csv')