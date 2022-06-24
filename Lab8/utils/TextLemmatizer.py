from pymorphy2 import MorphAnalyzer
from utils.tokenizers.WordTokenizer import WordTokenizer

class TextLemmatizer:
    def __init__(self, language, abbreviations = None):
        self.__morph = MorphAnalyzer(lang=language)
        self.__abbreviations = abbreviations


    def __call__(self, text):
        """
        Lemmatizes a text.
        """
        return [self.__morph.parse(word)[0].normal_form
            for word in WordTokenizer(text).tokenize(self.__abbreviations)]
