class SpacyTokenizer:
    """ Tool for tokenize powered by spacy module
    """

    def __init__(self, lang: str, disable=['parser', 'tagger', 'ner']):
        import spacy
        """ Initialize the language type for token

        Note: Are you looking for `tokenizer = torchtext.data.utils.get_tokenizer('basic_english')`?
        
        Args:
            lang (str): `en_core_web_sm` for English
            see https://spacy.io/usage/models, https://spacy.io/api/top-level/#spacy.load
        
        Examples:
            >>> tokenizer = SpacyTokenizer('en_core_web_sm')
            >>> text = 'SP500 rise to 32,000 (rise 12%).\n Hi, I am a boy!'
            >>> text = ['SP500 rise to 32,000 (rise 12%).', 'Hi, I am a boy!']
            >>> tokenizer.tokenize(text)
        """
        self._nlp = spacy.load(lang, disable=disable)

    def tokenize(self, text: str) -> list:
        """
        Args:
            text (str or List[str]): if provided `str`, will split it by '\n' than tokenize each line.
          
        Returns:
            List[str]: tokenized text
        """
        if not isinstance(text, list):
            lines = text.splitlines()
        else:
            lines = text

        doc = [[token.text for token
                in self._nlp.tokenizer(text.strip())] for text in lines]

        return doc
