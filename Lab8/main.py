import config
import pandas as pd
from helpers import \
    cleanse_column, \
    dump_json, \
    get_bag_of_words, \
    lemmatize_column, \
    tokenize_column, \
    read_list, \
    vectorize_top_words
from pandas import read_csv

pd.set_option('display.max_colwidth', None)

abbreviations = read_list(config.ABBREVIATIONS_PATH)
stop_words = read_list(config.STOP_WORDS_PATH)

with open(config.STOP_WORDS_PATH, encoding="utf-8") as f:
    stop_words = f.read().splitlines()

ukr_news = read_csv(config.TEXT_PATH, encoding='utf-8')

ukr_news["Body"] = cleanse_column(ukr_news, "Body", stop_words, abbreviations)

print(tokenize_column(ukr_news.head(), "Body", "sentence", abbreviations))
print(tokenize_column(ukr_news.head(), "Body", "word", abbreviations))

print(lemmatize_column(ukr_news.head(), "Body", "uk", abbreviations))

tfidf_dict = vectorize_top_words(ukr_news["Body"], 10, "uk", stop_words, abbreviations)

dump_json(config.TFIDF_DICT_PATH, tfidf_dict, indent=4, ensure_ascii=False)

bag_dict = get_bag_of_words(ukr_news["Body"], "uk", abbreviations)
dump_json(config.BAG_OF_WORDS_DICT_PATH, bag_dict, indent=4, ensure_ascii=False)
