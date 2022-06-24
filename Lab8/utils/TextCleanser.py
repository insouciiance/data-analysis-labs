import string

class TextCleanser:
    def __init__(self, text):
        self.__text = text


    def cleanse(self, stop_words : list = None, abbreviations : list = None):
        """
        Cleanses the text by removing punctuation, numbers, and stop words.
        """
        #Lower text
        text = self.__text.lower()

        # Remove abbreviations
        if abbreviations is not None:
            for abbreviation in abbreviations:
                text = text.replace(abbreviation, "")

        # Remove punctuation, apart from dots
        text = text.translate(str.maketrans("", "",
            string.punctuation
                .replace(".", "")
                .replace("?", "")
                .replace("!", "")))

        # Remove numbers
        text = "".join(ch for ch in text if not ch.isdigit())

        # Remove stop words
        if stop_words is not None:
            text = " ".join(word for word in text.split() if word not in stop_words)

        return text
