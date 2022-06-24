from utils.tokenizers.ITextTokenizer import ITextTokenizer
from nltk.tokenize import RegexpTokenizer

class WordTokenizer (ITextTokenizer):
    def __init__(self, text):
        super().__init__(text)


    def tokenize(self, abbreviations : list = None):
        tokenizer = RegexpTokenizer(r'\w+')
        
        if abbreviations is not None:
            tokenizer.abbrev_types = abbreviations
        
        return tokenizer.tokenize(self.text)
