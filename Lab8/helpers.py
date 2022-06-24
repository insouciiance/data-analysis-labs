import json
from collections import Counter
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils.TextCleanser import TextCleanser
from utils.tokenizers.WordTokenizer import WordTokenizer
from utils.tokenizers.SentenceTokenizer import SentenceTokenizer
from utils.TextLemmatizer import TextLemmatizer


def cleanse_column(
    dataset : DataFrame,
    column : str,
    stop_words : list,
    abbreviations : list):
    """
    Cleanses a column of text by removing stop words.
    """
    return dataset[column].apply(lambda x: TextCleanser(x).cleanse(stop_words, abbreviations))


def tokenize_column(
    dataset : DataFrame,
    column : str,
    tokenizer_type : str,
    abbreviations : list):
    """
    Tokenizes a column of text by sentence or word.
    """
    tokenizer = None
    if tokenizer_type == "word":
        tokenizer = WordTokenizer
    elif tokenizer_type == "sentence":
        tokenizer = SentenceTokenizer

    return dataset[column].apply(lambda x: tokenizer(x).tokenize(abbreviations))


def lemmatize_column(
    dataset : DataFrame,
    column : str,
    language : str,
    abbreviations : list):
    """
    Lemmatizes a column of text.
    """
    return dataset[column].apply(lambda x: TextLemmatizer(language, abbreviations)(x))


def read_list(path : str, encoding : str = "utf-8"):
    """
    Reads a list from a file.
    """
    with open(path, encoding=encoding) as f:
        return f.read().splitlines()


def vectorize_top_words(
    column : DataFrame,
    n : int,
    language : str,
    stop_words : list,
    abbreviations : list):
    """
    Vectorizes a text by top n words.
    """
    text = " ".join(column.apply(lambda x : x))
    words = TextLemmatizer(language, abbreviations)(text)

    top_words = Counter(words).most_common(n)

    words = list(set(words))
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        tokenizer=TextLemmatizer(language, abbreviations))
    
    vectorized_words = vectorizer.fit_transform(column).toarray()

    tfidf_dict = {}

    for i in range(0, len(column)):
        tfidf_dict[i] = {}
        for word in words:
            if word in [tuple[0] for tuple in top_words]:
                tfidf_dict[i][word] = vectorized_words[i][vectorizer.vocabulary_.get(word, 0)]

    return tfidf_dict


def get_bag_of_words(column : DataFrame):
    """
    Vectorizes a text by bag of words.
    """
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(column).toarray()
    
    bag_dict = {}

    for i in range(0, 5): # len(column) would generate around 250MB json
        bag_dict[i] = {}
        for key, value in vectorizer.vocabulary_.items():
            bag_dict[i][key] = int(bag_of_words[i][value])

    return bag_dict


def dump_json(filename : str, data : dict, **kwargs):
    """
    Dumps a dictionary to a json file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)
