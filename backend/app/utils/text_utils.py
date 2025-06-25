import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import unicodedata



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
        transformed_text, _ = TextUtils.get_transformed_text_with_mapping(text_with_underscores, 
                                                          stop_words, lema, stem)
        transformed_text = transformed_text.replace(' ', '_')
        
        return transformed_text

    
    @staticmethod
    def get_transformed_text_with_mapping(
            text: str,
            stop_words: tuple[str] = (), 
            lema: bool = True, 
            stem: bool = False
            ) -> tuple[str, list[tuple[int, int, str, str]]]:
        """
        Apply transformations to the text and map raw words to processed words with their positions in the text.

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
        raw_to_processed_map_list : list[tuple[int, int, str, str]]
            A list of mapping tuples from raw text index positions to its raw and processed word respectively.
        """
        # Tokenize the text into raw tokens
        raw_tokens = nltk.word_tokenize(text)

        # Split hyphens and underscores, then remove punctuation and special characters, to preprocess tokens
        raw_splitted_words = TextUtils.split_strings_with_hyphen_and_underscore(raw_tokens)
        filtered_words = TextUtils.convert_to_lowercase(raw_splitted_words)
        filtered_words = TextUtils.replace_accented_vowels(filtered_words)
        filtered_words = TextUtils.remove_special_characters(filtered_words)
        filtered_words = TextUtils.remove_stopwords_and_only_digit_words(filtered_words, stop_words)  # Remove stopwords
        
        # Apply lemmatization
        if lema:
            filtered_words = TextUtils.do_lemmatization(filtered_words)
        
        # Apply stemming
        if stem:
            filtered_words = TextUtils.do_stemming(filtered_words)
        
        # Join the tokens back into a single string
        transformed_text = TextUtils.join_tokens(filtered_words)
        
        # Initialize list of mapping tuples
        raw_to_processed_map_list = TextUtils.get_raw_to_processed_map_list(text, raw_splitted_words, filtered_words)
        
        return transformed_text, raw_to_processed_map_list
    
    
    @staticmethod
    def split_strings_with_hyphen_and_underscore(tokens: list[str]) -> list[str]:
        """
        Split strings in the given list that contain hyphens or underscores into separate tokens.

        This function takes a list of strings as input, where some strings may contain hyphens or underscores.
        It splits these strings into separate tokens at the hyphens or underscores and returns a new list containing
        all the individual tokens.

        Parameters:
        tokens (list[str]): A list of strings where some strings may contain hyphens or underscores.

        Returns:
        list[str]: A list of strings where each string represents a token.
            If a string in the input list contained hyphens or underscores, it has been split into separate tokens.
        """
        result = []
        for s in tokens:
            if '-' in s or '_' in s:
                # Use regex to split by both '-' and '_'
                parts = re.split(r'[-_]', s)
                result.extend(parts)
            else:
                result.append(s)
        return result
    
    
    @staticmethod
    def convert_to_lowercase(tokens: list[str]) -> list[str]:
        """
        Convert a list of tokens to lowercase.

        Parameters
        ----------
        tokens : list[str]
            A list of tokens to be converted to lowercase.

        Returns
        -------
        list[str]
            A list of tokens where each token is in lowercase.
        """
        return [token.lower() for token in tokens]
    
    
    @staticmethod
    def replace_accented_vowels(tokens: list[str]) -> list[str]:
        """
        Replace accented vowels in the given list of strings (tokens) with their non-accented counterparts.

        This function takes a list of strings as input and replaces accented vowels (á, é, í, ó, ú, ñ, Ñ)
        with their non-accented counterparts (a, e, i, o, u, n, N) in each string. It uses the `unicodedata` module
        to normalize the text and remove accents and diacritical marks.

        Parameters:
        tokens (List[str]): The input list of strings containing accented vowels.

        Returns:
        list[str]: The output list of strings with accented vowels replaced by their non-accented counterparts.
        """
        def replace_in_text(text: str) -> str:
            # Normalize the text to separate special characters
            normalized_text = unicodedata.normalize('NFKD', text)
            
            # Remove accents and diacritical marks
            text_without_diacritics = ''.join(
                c for c in normalized_text if unicodedata.category(c) != 'Mn'
            )
            
            # Replace 'ñ' and 'Ñ' with 'n' and 'N'
            final_text = text_without_diacritics.replace('ñ', 'n')
            
            return final_text

        # Apply the replacement to each string in the list
        return [replace_in_text(token) for token in tokens]
    
    
    @staticmethod
    def remove_special_characters(tokens: list[str]) -> list[str]:
        """
        Remove special characters and leading/trailing whitespace from a list of tokens.

        This function takes a list of tokens as input, where each token is a string.
        It applies a regular expression to remove special characters (non-alphanumeric and non-whitespace characters)
        from each token. It also removes leading and trailing whitespace from each token.
        The cleaned tokens are then returned as a new list.

        Parameters:
        tokens (list[str]): A list of strings where each string represents a token.

        Returns:
        list[str]: A list of strings where each string represents a token that has been cleaned.
            The cleaned tokens have removed special characters and leading/trailing whitespace.
        """
        cleaned_list = []
        for t in tokens:
            cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', t)  # Delete special characters
            cleaned_string = cleaned_string.strip()  # Remove leading and trailing whitespace
            cleaned_list.append(cleaned_string)
        return cleaned_list
    
    
    @staticmethod
    def remove_stopwords_and_only_digit_words(
        tokens: list[str], 
        stop_words: tuple[str]
        ) -> list[str]:
        """
        Remove stopwords and words that only contain digits from a list of tokens.

        This function takes a list of tokens and a tuple of stop words as input.
        It filters out stopwords and words that only contain digits from the list of tokens.
        The filtered list of tokens is then returned.

        Parameters:
        tokens (list[str]): A list of strings where each string represents a token.
        stop_words (tuple[str]): A tuple of strings where each string is a stop word.

        Returns:
        list[str]: A list of strings where each string represents a token that has been filtered.
            The filtered tokens do not contain stopwords or words that only contain digits.
        """
        filtered_words = []
        
        for word in tokens:
            if word in stop_words or (word and word.isdigit()):
                cleaned_word = ""
            else:
                cleaned_word = word
            filtered_words.append(cleaned_word)
        
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
    
    
    @staticmethod
    def join_tokens(tokens: list[str]) -> str:
        """
        Join a list of tokens into a single string, ignoring empty tokens.

        Parameters
        ----------
        tokens : list[str]
            A list of tokens to be joined into a single string.

        Returns
        -------
        str
            A single string containing the joined tokens, separated by spaces.
        """
        return ' '.join([w for w in tokens if w != ''])
    
    
    @staticmethod
    def get_raw_to_processed_map_list(
            text: str, 
            raw_splitted_words: list[str], 
            filtered_words: list[str]
            ) -> list[tuple[int, int, str, str]]:
        """
        Get a list of mapping tuples from raw text positions to processed words.
        This function takes the raw text, raw splitted words, and filtered words as input,
        and returns a list of mapping tuples from raw text positions to processed words,
        with format: [ (first_idx, last_idx, raw_word, processed_word), ... ]
        
        Parameters:
        text (str): The raw text containing the words to be processed.
        raw_splitted_words (list[str]): A list of raw splitted words extracted from the raw text.
        filtered_words (list[str]): A list of processed words extracted from the raw text.
        
        Returns:
        list[tuple[int, int, str, str]]: A list of mapping tuples from raw text index positions to 
        its raw and processed word respectively.
        """
        # Initialize list of mapping tuples
        raw_to_processed_map_list: list[tuple[int, int, str, str]] = []
        
        # Map raw words to processed words with index positions
        current_position = 0  # Tracks the current position in the raw text
        for raw_word, processed_word in zip(raw_splitted_words, filtered_words):
            # Find the start and end indices of the current raw word
            start_idx = text.find(raw_word, current_position)
            if start_idx == -1:
                continue  # Skip if the word is not found
            end_idx = start_idx + len(raw_word) - 1

            # Add the elements to a new item in the mapping list
            if processed_word:
                raw_to_processed_map_list.append((start_idx, end_idx, raw_word, processed_word))

            # Move the current position forward to avoid finding the same word again
            current_position = end_idx + 1
            
        return raw_to_processed_map_list