import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

class TweetTextPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')

    def preprocess(self, tweet_texts):
        tweet_texts = self.__convert_to_lowercase(tweet_texts)
        tweet_texts = self.__clean_text(tweet_texts)
        tweet_texts = self.__tokenize(tweet_texts)
        tweet_texts = self.__remove_stopwords(tweet_texts)
        tweet_texts = self.__stem(tweet_texts)
        return tweet_texts

    def __convert_to_lowercase(self, texts):
        return texts.str.lower()

    def __clean_text(self, texts):
        texts = self.__remove_urls(texts)
        texts = self.__remove_mentions(texts)
        texts = self.__remove_nonword_and_nonwhitespace(texts)
        texts = self.__remove_digits(texts)
        return texts

    def __tokenize(self, texts):
        return texts.apply(word_tokenize)

    def __remove_stopwords(self, texts):
        stopword_list = stopwords.words('english')
        return texts.apply(self.__filter_stopwords, stopwords=stopword_list)

    def __stem(self, texts):
        self.stemmer = PorterStemmer()
        return texts.apply(self.__stem_text)

    def __remove_urls(self, texts):
        return texts.replace('{link}', '', regex=False)

    def __remove_mentions(self, texts):
        return texts.replace('@mention', '', regex=False)

    def __remove_nonword_and_nonwhitespace(self, texts):
        return texts.replace(to_replace=r'[^\w\s]', value='', regex=True)

    def __remove_digits(self, texts):
        return texts.replace(to_replace=r'\d', value='', regex=True)

    def __filter_stopwords(self, text, stopwords):
        return [word for word in text if word not in stopwords]

    def __stem_text(self, text):
        return [self.stemmer.stem(word) for word in text]