#import srex_new
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


class Ranking:
    
    def __init__(self, query_text: str, reference_terms: list[str], nr_search_results: int = 10, ranking_weight_type: str = 'linear', 
                 stop_words: list[str] = [], lema: bool = True, stem: str = False):
        self.__query_text = query_text
        self.__reference_terms = reference_terms
        self.__nr_search_results = nr_search_results
        self.__ranking_weight_type = ranking_weight_type   #Type of weighting to be applied (it can be: 'none', 'linear' or 'inverse')
        self.__stop_words = stop_words
        self.__lema = lema
        self.__stem = stem

        self.__documents: list[Document] = []
        self.__graph = Graph()

        results = self.__get_ieee_explore_ranking()
        self.__calculate_ranking_as_weighted_documents_and_do_text_transformations(results)
        self.__do_text_transformations_by_string_in_list()
        processed_sentences_list_with_refterms = self.__delete_sentences_without_refterms(processed_sentences_list)


    def get_ranking(self) -> list:
        return self.__documents 


    def __get_ieee_explore_ranking(self) -> list[dict]:
        """
        Get a ranking of articles from IEEE-Xplore.

        Returns
        -------
        results : list[dict]
            A list of articles
        """
        xplore_id = '6g7w4kfgteeqvy2jur3ak9mn'
        query = XPLORE(xplore_id)
        query.outputDataFormat='object'
        query.maximumResults(self.__nr_search_results)
        query.queryText(self.__query_text)
        data = query.callAPI()
        results = data.get('articles', [{}])
        
        return results


    def __calculate_ranking_as_weighted_documents_and_do_text_transformations(
            self,
            results: list[dict],
            ):
        """
        Transform the ranking array in a list of strings associated with their weights.
        if weighted <> none : the document will be weighted depending on its position in the ranking

        Parameters
        ----------
        results : list[dict]
            Array of documents (articles)
        """
        results_size = len(results)
        for index, article in enumerate(results):
            # Calculate the weight depending on the argument value and the position of the document in the ranking
            _weight = self.__calculate_weight(self.__ranking_weight_type, results_size, index)
            _abstract = article.get('abstract', "")
            _title = article.get('title', "")
            _doc_id = article.get('article_number', "1")
            _ranking_pos = article.get('rank', 1)
            new_doc = Document(abstract=_abstract, title=_title, doc_id=_doc_id, weight=_weight, ranking_position=_ranking_pos)
            new_doc.do_text_transformations_by_sentence(self.__stop_words, self.__lema, self.__stem)
            self.__documents.append(new_doc)


    def __calculate_weight(
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


    def __do_text_transformations_by_string_in_list(self) -> list[str]:
        """
        Apply text transformations to a list of strings. Remove stopwords, punctuation, stemming, lematization.

        Returns
        -------
        processed_string_list : list[str]
            A list of transformed strings.
        """
        processed_string_list = []
        for term in self.__reference_terms:
            processed_string_list.append(self.do_text_transformations(term, self.__stop_words, self.__lema, self.__stem))

        self.__reference_terms = processed_string_list


    def __delete_sentences_without_refterms(
            self,
            documents_list: list[dict]
            ) -> list[dict]:
        """
        Delete sentences from a list of sentences by document that do not contain a reference term

        Parameters
        ----------
        documents_list: list[dict]
            List of dictionaries (articles), each one contains a list (array) of sentences
        
        Returns
        -------
        documents_list_with_refterms : list[dict]
            List of documents that exclusively contain a reference term in its list of sentences
        """
        for document in documents_list:
            sentences_with_refterms = []
            for sentence in document['text']:
                for term in self.__reference_terms:
                    if term in sentence:
                        sentences_with_refterms.append(sentence)
                        break
            document['text'] = sentences_with_refterms

        return documents_list



class Document:
    
    def __init__(self, abstract: str = "", title: str = "", doc_id: str = "1", weight: float = 1, ranking_position: int = 1):
        self.__abstract = abstract
        self.__title = title
        self.__doc_id = doc_id
        self.__weight = weight
        self.__ranking_position = ranking_position
        self.__sentences: list[Sentence] = []
        self.__graph = Graph()

        self.__calculate_sentences_list_from_documents()
    

    def get_abstract(self) -> str:
        return self.__abstract
    

    def get_title(self) -> str:
        return self.__title
    

    def get_doc_id(self) -> str:
        return self.__doc_id


    def get_ieee_explore_article(
            self,
            parameter: str, 
            value: str
            ):
        """
        Get an article from IEEE-Xplore.
        
        Parameters
        ----------
        parameter
            Parameter used to search the article (e.g. 'article_number')
        value
            Value of the parameter used to search the article (e.g. '8600704')

        """
        xplore_id = '6g7w4kfgteeqvy2jur3ak9mn'
        query = XPLORE(xplore_id)
        query.outputDataFormat='object'
        query.addParameter(parameter, value)
        data = query.callAPI()
        self.__abstract = data.get('articles', [{}])[0].get('abstract', "")
        self.__title = data.get('articles', [{}])[0].get('title', "")
        self.__doc_id = data.get('articles', [{}])[0].get('article_number', "1")


    def __calculate_sentences_list_from_documents(
            self,
            ) -> list[dict]:
        """
        Transform the text of each document in the input list into a list of sentences. Split the text by dots.
        """
        sentence_list = [self.__title]
        abstract_sentence_list = self.__abstract.split('. ')
        sentence_list.extend(abstract_sentence_list)
        for index, sentence_str in enumerate(sentence_list):
            sentence_obj = Sentence(raw_text=sentence_str, position_in_doc=index)
            self.__sentences.append(sentence_obj)
    

    def do_text_transformations_by_sentence(
            self,
            stop_words_list: list[str], 
            lema: bool,
            stem: bool
            ):
        """
        Apply some text transformations to each sentence of the document. Remove stopwords, punctuation, stemming, lematization

        Parameters
        ----------
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        """
        for sentence in self.__sentences:
            sentence.do_text_transformations(stop_words_list, lema, stem)



class Sentence:
    
    def __init__(self, raw_text: str, position_in_doc: int):
        self.__raw_text = raw_text
        self.__position_in_doc = position_in_doc
        self.__processed_text: str = ""


    def get_raw_text(self) -> str:
        return self.__raw_text


    def get_position_in_doc(self) -> int:
        return self.__position_in_doc


    def get_processed_text(self) -> str:
        return self.__processed_text
    

    def do_text_transformations(
            self,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ):
        """
        Apply some text transformations to the sentence. Remove stopwords, punctuation, stemming, lematization.

        Parameters
        ----------
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        """
        sentence = self.__raw_text

        # Low the string
        sentence = sentence.lower()
        
        # Remove puntuation
        tokens = nltk.word_tokenize(sentence)
        filtered_sentence = [token for token in tokens if token.isalnum()]
        
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
        
        self.__processed_text = final_string



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
