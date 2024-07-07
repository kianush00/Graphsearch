import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



class TextUtils:

    @staticmethod
    def get_transformed_text_if_it_has_underscores(
            text_with_underscores: str,
            stop_words_list: list[str] = [], 
            lema: bool = True, 
            stem: bool = False
            ) -> str:
        """
        Apply some transformations to the text with underscores. First the underscores 
        are replaced, then the text is transformed, and finally the underscores are 
        reinserted into the text

        Parameters
        ----------
        text_with_underscores : str
            Text with underscores to be transformed
        stop_words_list : list[str]
            List of stop words to be removed from the text provided
        lema : bool, optional
            If True, lemmatization is applied
        stem : bool, optional
            If True, stemming is applied
        
        Returns
        -------
        transformed_text : str
            The transformed text
        """
        text_with_underscores = text_with_underscores.replace('_', ' ')
        transformed_text = TextUtils.get_transformed_text(text_with_underscores, 
                                                          stop_words_list, lema, stem)
        transformed_text = transformed_text.replace(' ', '_')
        
        return transformed_text

    
    @staticmethod
    def get_transformed_text(
            text: str,
            stop_words: list[str] = [], 
            lema: bool = True, 
            stem: bool = False
            ) -> str:
        """
        Apply some transformations to the text. Lower the text, tokenize, remove punctuation, 
        stopwords, finally do stemming and lemmatization if specified.

        Parameters
        ----------
        text : str
            Text to be transformed
        stop_words : list[str]
            List of stop words to be removed from the text provided
        lema : bool, optional
            If True, lemmatization is applied
        stem : bool, optional
            If True, stemming is applied
        
        Returns
        -------
        transformed_text : str
            The transformed text
        """
        # Convert the string to lowercase
        lower_text = text.lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(lower_text)

        # Remove punctuation
        filtered_words = TextUtils.remove_special_characters(tokens)
        
        # Remove stopwords
        filtered_words = TextUtils.remove_stopwords(filtered_words, stop_words)
        
        # Apply lemmatization
        if lema:
            filtered_words = TextUtils.do_lemmatization(filtered_words)
        
        # Apply stemming
        if stem:
            filtered_words = TextUtils.do_stemming(filtered_words)
        
        # Join the tokens back into a single string
        transformed_text = ' '.join(filtered_words)
        
        return transformed_text
    

    @staticmethod
    def remove_special_characters(tokens: list[str]) -> list[str]:
        """
        Remove special characters from a list of tokens.
        """
        filtered_words = [token for token in tokens if token.isalnum()]
        return filtered_words
    

    @staticmethod
    def remove_stopwords(
        tokens: list[str], 
        stop_words: list[str]
        ) -> list[str]:
        """
        Remove specified stop words from a list of tokens.
        """
        if len(stop_words) > 0:
            filtered_words = [word for word in tokens if word not in stop_words]
        else:
            filtered_words = tokens
        return filtered_words
    

    @staticmethod
    def do_lemmatization(tokens: list[str]) -> list[str]:
        """
        Apply lemmatization (reduce a word to its lemma root form) to a list of tokens.
        E.g.  running -> run  ;  better -> good  ;  went -> go
        """
        lemmatizer = WordNetLemmatizer()
        filtered_words = [lemmatizer.lemmatize(word) for word in tokens]
        return filtered_words
    

    @staticmethod
    def do_stemming(tokens: list[str]) -> list[str]:
        """
        Apply stemming (reduce a word to its stem root form) to a list of tokens.
        E.g.  running -> run  ;  better -> bett  ;  went -> went
        """
        stemmer = PorterStemmer()
        filtered_words = [stemmer.stem(word) for word in tokens]
        return filtered_words