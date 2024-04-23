import srex_new
import operator
import numpy as np
import math
from collections import defaultdict
#import pandas as pd
from nltk.stem import PorterStemmer as st #Stemmer
from textblob import Word #Lemmatize
from graphviz import Graph
import nltk
from functools import reduce
from xploreapi import XPLORE
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from numpy import dot

class Text:
    
    def __init__(self, raw_text: str):
        self.__raw_text = raw_text
        self.__processed_text: str = ""


    def get_raw_text(self) -> str:
        return self.__raw_text
    

    def get_processed_text(self) -> str:
        return self.__processed_text



class Ranking:
    
    def __init__(self, query_text: str, reference_terms: list[str], nr_search_results: int = 10, stop_words: list[str] = [], 
                 ranking_weight_type: str = 'linear', lema: bool = True, stem: str = False):
        self.__query_text = query_text
        self.__reference_terms = reference_terms
        self.__nr_search_results = nr_search_results
        self.__stop_words = stop_words
        self.__ranking_weight_type = ranking_weight_type   # it can be: 'none', 'linear' or 'inverse'
        self.__lema = lema
        self.__stem = stem

        self.__xplore_ID = '6g7w4kfgteeqvy2jur3ak9mn'
        self.__documents: list[Document] = []
        self.__graph = Graph()

        results = self.__get_ieee_explore_ranking(self.__query_text, self.__nr_search_results)
        list_of_documents = self.__get_ranking_as_weighted_documents(results, self.__ranking_weight_type)
        sentences_list = self.__get_sentences_list_from_documents(list_of_documents)
        processed_sentences_list = self.__do_text_transformations_by_document(sentences_list, self.__stop_words, self.__lema, self.__stem)
        self.__reference_terms = self.__do_text_transformations_by_string_in_list(self.__reference_terms, self.__stop_words, self.__lema, self.__stem)
        processed_sentences_list_with_refterms = self.__delete_sentences_without_refterms(processed_sentences_list, self.__reference_terms)


    def get_ranking(self) -> list:
        return self.__documents 


    def __get_ieee_explore_ranking(
            self,
            query_text: str, 
            max_results: int
            ) -> list[dict]:
        """
        Get a ranking of articles from IEEE-Xplore.

        Parameters
        ----------
        query_text : str
            Text used to search the articles  (e.g. 'internet of things')
        max_results : int
            Maximum number of results to be returned

        Returns
        -------
        results : list[dict]
            A list of articles
        """
        query = XPLORE(self.__xplore_ID)
        query.outputDataFormat='object'
        query.maximumResults(max_results)
        query.queryText(query_text)
        data = query.callAPI()
        results = data['articles']
        
        return results


    def __get_ranking_as_weighted_documents(
            self,
            results: list[dict], 
            weighted: str = 'linear'
            ) -> list[dict]:
        """
        Transform the ranking array in a list of strings associated with their weights.
        if weighted <> none : the document will be weighted depending on its position in the ranking

        Parameters
        ----------
        results : list[dict]
            Array of documents (articles)
        weighted : str
            Type of weighting to be applied (it can be: 'none', 'linear' or 'inverse')
        
        Returns
        -------
        ranking : list[dict]
            A list of dictionaries (documents), each dictionary contains the title and the abstract of an article as a 
            text (string), along with its weight (float)
        """
        ranking = []
        results_size = len(results)
        for index, article in enumerate(results):
            # Calculate the weight depending on the argument value and the position of the document in the ranking
            factor = self.__calculate_factor(weighted, results_size, index)
            new_text = article['title'] + '. ' + article['abstract']
            ranking.append({'text': new_text, 'weight': factor})
        return ranking


    def __calculate_factor(
            self,
            weighted: str, 
            results_size: int, 
            index: int
            ) -> float:
        """
        Calculate a weight factor depending on the argument value ('linear' or 'inverse'), and 
        the position of the document in the ranking.

        Parameters
        ----------
        weighted : str
            The type of weighting to be applied. Can be 'linear', 'inverse', or any other value.
        results_size : int
            The total number of results/documents.
        index : int
            The position of the document in the ranking.

        Returns
        -------
        factor : float
            The calculated weight factor.
        """
        if (weighted=='linear'):
            factor = float((results_size - (index * 0.5)) / results_size)
        elif (weighted=='inverse'):
            factor = float(math.ceil(results_size / ((index * 0.5) + 1)) / results_size)
        else:
            factor = 1.0

        return factor


    def __get_sentences_list_from_documents(
            self,
            list_of_documents: list[dict]
            ) -> list[dict]:
        """
        Transform the text of each document in the input list into a list of sentences. Split the text by dots.

        Parameters
        ----------
        list_of_documents : list[dict]
            A list of dictionaries representing documents, where each dictionary has a 'text' key containing the document text.

        Returns
        -------
        list_of_documents_with_sentences_list : list[dict]
            A list of dictionaries representing documents, where each dictionary has a 'text' key containing a list of sentences.
        """
        for document in list_of_documents:
            document['text'] = document['text'].split('. ')
        return list_of_documents
    

    def __do_text_transformations_by_document(
        self,
        documents_list: list[dict],
        stop_words_list: list[str], 
        lema: bool = True, 
        stem: bool = True
        ) -> list[dict]:
        """
        Apply some text transformations to a list of documents. Remove stopwords, punctuation, stemming, lematization

        Parameters
        ----------
        string_list : list[str]
            A list of documents to be transformed.
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied

        Returns
        -------
        transformed_documents_list : list[dict]
            A list of transformed documents.
        """
        for document in documents_list:
            document["text"] = list(map(lambda sentence: self.__text_transformations(sentence, stop_words_list, lema, stem), document["text"]))

        return documents_list


    def __do_text_transformations_by_string_in_list(
            self,
            string_list: list[str],
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> list[str]:
        """
        Apply text transformations to a list of strings. Remove stopwords, punctuation, stemming, lematization.

        Parameters
        ----------
        string_list : list[str]
            A list of strings to be transformed.
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied

        Returns
        -------
        processed_string_list : list[str]
            A list of transformed strings.
        """
        processed_string_list = []
        for term in string_list:
            processed_string_list.append(self.__text_transformations(term, stop_words_list, lema, stem))

        return processed_string_list


    def __text_transformations(
            self,
            sentence: str, 
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> str:
        """
        Apply some text transformations to a sentence. Remove stopwords, punctuation, stemming, lematization.

        Parameters
        ----------
        sentence : str
            String with the sentence to be transformed
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        
        Returns
        -------
        final_string : str
            The transformed sentence
        """
        
        # Low the string
        sentence = sentence.lower()
        
        # Remove puntuation
        tokens = nltk.word_tokenize(sentence)
        filtered_sentence = [tk for tk in tokens if tk.isalnum()]
        
        # Remove Stopwords
        if(len(stop_words_list)>0):
            filtered_sentence = list(filter(lambda word_of_sentence: (word_of_sentence not in stop_words_list), filtered_sentence))
        
        # Apply lematization
        if(lema):
            filtered_sentence = list(map(lambda word_filtered_sentence: Word(word_filtered_sentence).lemmatize(), filtered_sentence))
        
        # Stemmer
        if(stem):
            filtered_sentence = list(map(lambda word: st.stem(word), filtered_sentence))
        
        final_string = ' ' . join(map(str, filtered_sentence))
        
        return final_string


    def __delete_sentences_without_refterms(
            self,
            documents_list: list[dict],
            reference_terms: list[str], 
            ) -> list[dict]:
        """
        Delete sentences from a list of sentences by document that do not contain a reference term

        Parameters
        ----------
        documents_list: list[dict]
            List of dictionaries (articles), each one contains a list (array) of sentences
        reference_terms: list[str]
            List of reference terms that must be contained in the sentences
        
        Returns
        -------
        documents_list_with_refterms : list[dict]
            List of documents that exclusively contain a reference term in its list of sentences
        """
        for document in documents_list:
            sentences_with_refterms = []
            for sentence in document['text']:
                for term in reference_terms:
                    if term in sentence:
                        sentences_with_refterms.append(sentence)
                        break
            document['text'] = sentences_with_refterms

        return documents_list



class Document(Text):
    
    def __init__(self, raw_text: str, doc_id: int, weight: float, ranking_position: int, title: str):
        
        super().__init__(raw_text)
        self.__doc_id = doc_id
        self.__weight = weight
        self.__ranking_position = ranking_position
        self.__title = title
        self.__sentences: list[Sentence] = []
        self.__graph = Graph()



class Sentence(Text):
    
    def __init__(self, raw_text: str, position_in_doc: int):
        super().__init__(raw_text)
        self.__position_in_doc = position_in_doc


    def get_position_in_doc(self) -> int:
        return self.__position_in_doc



class Graph:
        
    def __init__(self, reference_terms: list[str], nr_of_graph_terms: int = 5, limit_distance: int = 4, include_ref_term: bool = True, 
                 format_adjac_refterms: bool = True):

        #self.nodes = {}
        self.__reference_terms = reference_terms
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_reference_term = include_ref_term
        self.__format_adjacent_refterms = format_adjac_refterms

        self.__vecinity_matrix = []
        self.__most_freq_dist_list = []
        

    def get_vecinity_matrix(self) -> list[dict]:
        return self.__vecinity_matrix
    

    def get_most_freq_dist_list(self) -> list[dict]:
        return self.__most_freq_dist_list
