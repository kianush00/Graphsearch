import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



class TextUtils:

    @staticmethod
    def get_transformed_text_if_it_has_underscores(
            text_with_underscores: str,
            stop_words: tuple[str] = (), 
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
        stop_words : tuple[str]
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
                                                          stop_words, lema, stem)
        transformed_text = transformed_text.replace(' ', '_')
        
        return transformed_text

    
    @staticmethod
    def get_transformed_text(
            text: str,
            stop_words: tuple[str] = (), 
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
        stop_words : tuple[str]
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

        Parameters:
        tokens (list[str]): A list of strings where each string represents a token.

        Returns:
        list[str]: A list of strings where special characters have been removed.
        Each string in the returned list is a token that does not contain any special characters.
        """
        filtered_words = [token for token in tokens if token.isalnum()]
        return filtered_words
    

    @staticmethod
    def remove_stopwords(
        tokens: list[str], 
        stop_words: tuple[str]
        ) -> list[str]:
        """
        Remove specified stop words from a list of tokens.

        This function takes a list of tokens and a tuple of stop words as input.
        It filters out the stop words from the list of tokens and returns a new list
        containing only the tokens that are not stop words.

        Parameters:
        tokens (list[str]): A list of strings where each string represents a token.
        stop_words (tuple[str]): A tuple of strings representing stop words.

        Returns:
        list[str]: A list of strings where each string represents a token that is not a stop word.
        """
        filtered_words = [word for word in tokens if word not in stop_words]
        return filtered_words
    

    @staticmethod
    def do_lemmatization(tokens: list[str]) -> list[str]:
        """
        Apply lemmatization (reduce a word to its lemma root form) to a list of tokens.
        Lemmatization is a natural language processing technique that aims to reduce words to their base or dictionary form.
        This function takes a list of tokens as input and applies lemmatization to each token.
        The lemmatized tokens are then returned as a new list.

        Parameters:
        tokens (list[str]): A list of strings where each string represents a token.

        Returns:
        list[str]: A list of strings where each string represents a lemmatized token.
        The lemmatized tokens are the base or dictionary form of the original words.
        """
        lemmatizer = WordNetLemmatizer()
        filtered_words = [lemmatizer.lemmatize(word) for word in tokens]
        return filtered_words
    

    @staticmethod
    def do_stemming(tokens: list[str]) -> list[str]:
        """
        Apply stemming (reduce a word to its stem root form) to a list of tokens.
        Stemming is a text normalization technique that reduces words to their base or root form.
        This function takes a list of tokens as input and applies stemming to each token.
        The stemmed tokens are then returned as a new list.

        Parameters:
        tokens (list[str]): A list of strings where each string represents a token.
            Tokens are typically individual words or terms extracted from a text.

        Returns:
        list[str]: A list of strings where each string represents a stemmed token.
            The stemmed tokens are the base or root form of the original words.
            For example, "running" will be stemmed to "run", "better" to "bett", and "went" to "went".
        """
        stemmer = PorterStemmer()
        filtered_words = [stemmer.stem(word) for word in tokens]
        return filtered_words