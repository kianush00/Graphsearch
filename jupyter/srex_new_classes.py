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
    
    def __init__(self, query_text: str, nr_search_results: int = 10, stop_words: list[str] = [], ranking_weight_type: str = 'linear', lema: bool = True, stem: str = False):
        self.__query_text = query_text
        self.__nr_search_results = nr_search_results
        self.__stop_words = stop_words
        self.__ranking_weight_type = ranking_weight_type   # it can be: 'none', 'linear' or 'inverse'
        self.__lema = lema
        self.__stem = stem

        self.__xplore_ID = '6g7w4kfgteeqvy2jur3ak9mn'
        self.__documents: list[Document] = []
        self.__graph = Graph()

        results = self.__get_ieee_explore_ranking(query_text, nr_search_results)
        list_of_documents = self.__get_ranking_as_weighted_documents(results, ranking_weight_type)
        sentences_list = self.__get_sentences_list_from_documents(list_of_documents)


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



class Document(Text):
    
    def __init__(self, raw_text: str, doc_id: int, weight: float, ranking_position: int, title: str, ranking: Ranking):
        
        super().__init__(raw_text)
        self.__doc_id = doc_id
        self.__weight = weight
        self.__ranking_position = ranking_position
        self.__title = title
        self.__ranking = ranking
        self.__sentences: list[Sentence] = []
        self.__graph = Graph()



class Sentence(Text):
    
    def __init__(self, raw_text: str, document: Document, position_in_doc: int):
        super().__init__(raw_text)
        self.__document = document
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
