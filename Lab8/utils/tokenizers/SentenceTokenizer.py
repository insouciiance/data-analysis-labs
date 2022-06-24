from utils.tokenizers.ITextTokenizer import ITextTokenizer
from nltk.tokenize import PunktSentenceTokenizer

class SentenceTokenizer (ITextTokenizer):
    def __init__(self, text):
        super().__init__(text)


    def tokenize(self, abbreviations : list = None):
        tokenizer = PunktSentenceTokenizer()
        
        if abbreviations is not None:
            tokenizer.abbrev_types = abbreviations
        
        return tokenizer.tokenize(self.text)
