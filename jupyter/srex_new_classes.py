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
from enum import Enum


class Operation(Enum):
    UNION = 1
    INTERSECTION = 2


class BooleanOperation:
    def __init__(self, operation: Operation, operands: list):
        self.__operation: Operation = operation   # e.g. Operation.UNION
        self.__operands: list[BooleanOperation | str] = operands


    def get_operation(self) -> Operation: 
        return self.__operation
    

    def get_operands_str_list(self) -> list[str]:
        try:
            return self.recursive_get_operands()
        except:
            return ["ErrorUnknownOperandsType"]
    

    def recursive_get_operands(self) -> list[str]:
        operands_str_list = []
        for operand in self.__operands:
            if type(operand) == str:
                if len(operand) > 0:
                    operands_str_list.append(operand)
                else:
                    raise TypeError("Empty string operand")
            elif type(operand) == BooleanOperation: 
                operands_str_list.extend(operand.recursive_get_operands())
            else:
                raise TypeError("Unknown operand type")
            
        return operands_str_list
    

    def do_text_transformations_to_operands(self,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> None:
        for index, operand in enumerate(self.__operands):
            if type(operand) == str:
                if len(operand) > 0:
                    sentence_from_operand = Sentence(operand)
                    self.__operands[index] = sentence_from_operand.get_transformed_sentence_str(stop_words_list, lema, stem)
            elif type(operand) == BooleanOperation: 
                operand.do_text_transformations_to_operands()

    
    def __str__(self) -> str:
        try:
            return self.recursive_str()
        except:
            return "ErrorUnknownOperandsType"
    

    def recursive_str(self) -> str:
        string = "("
        for index, operand in enumerate(self.__operands):
            #print(f"INICIO str: {string}, index: {index}, len: {len(self.__operands)-1}")
            if index < (len(self.__operands)-1):
                if self.__operation == Operation.UNION:
                    string += self.__get_operand_chain(operand) + "|"
                elif self.__operation == Operation.INTERSECTION:
                    string += self.__get_operand_chain(operand) + "&"
            else:
                string += self.__get_operand_chain(operand)
        #print(f"FINAL str: {string}, index: {index}, len: {len(self.__operands)-1}")
        string += ")"
        return string
    

    def __get_operand_chain(self, operand) -> str:
        if type(operand) == str:
            if len(operand) > 0:
                return operand
            else:
                raise TypeError("Empty string operand")
        elif type(operand) == BooleanOperation: 
            return operand.recursive_str()
        else:
            raise TypeError("Unknown operand type")



class Sentence:
    
    def __init__(self, raw_text: str, reference_terms: BooleanOperation | str = "", position_in_doc: int = 0):
        self.__raw_text = raw_text
        self.__reference_terms = reference_terms
        self.__position_in_doc = position_in_doc
        self.__processed_text: str = ""
        self.__term_positions_dict: defaultdict[str, list[int]] = defaultdict(list)
        self.__graphs: list[Graph] = self.__create_and_get_graph_list()


    def get_raw_text(self) -> str:
        return self.__raw_text
    

    def get_reference_terms(self) -> BooleanOperation | str:
        return self.__reference_terms


    def get_position_in_doc(self) -> int:
        return self.__position_in_doc


    def get_processed_text(self) -> str:
        return self.__processed_text
    

    def set_processed_text(self, value: str) -> None:
        self.__processed_text = value
    

    def get_term_positions_dict(self) -> defaultdict[str, list[int]]:
        return self.__term_positions_dict
    

    def get_graphs(self) -> list[Graph]:
        return self.__graphs


    def do_text_transformations_and_term_positions_dict(self,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> None:
        """
        Apply some text transformations to the sentence and calculate term positions to the sentence.

        Parameters
        ----------
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        """
        transformed_sentence_str = self.get_transformed_sentence_str(stop_words_list, lema, stem)
        operands_str_list = self.__reference_terms.get_operands_str_list()
        if any(ref_term in transformed_sentence_str for ref_term in operands_str_list):
            self.__processed_text = transformed_sentence_str
            self.__do_term_positions_dict()


    def get_transformed_sentence_str(self,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> str:
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
        
        Returns
        -------
        final_string : str
            The transformed sentence
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
        
        return final_string


    def __do_term_positions_dict(self) -> None:
        """
        Calculate a dictionary with the sentence's term positions.
        """
        vectorizer = CountVectorizer()
        vector = vectorizer.build_tokenizer()(self.__processed_text)
        sentence_positions_dict = defaultdict(list)
        for i in range(len(vector)):
            sentence_positions_dict[vector[i]].append(i)
        self.__term_positions_dict = sentence_positions_dict
    

    def __create_and_get_graph_list(self) -> list[Graph]:
        """
        Create a list of graphs associated with the sentence
        """
        graph_list = []
        if type(self.__reference_terms) == str:
            graph_list.append(Graph(reference_terms=self.__reference_terms))
        elif type(self.__reference_terms) == BooleanOperation: 
            for ref_term in self.__reference_terms.get_operands_str_list():
                graph_list.append(Graph(reference_terms=ref_term))
        
        return graph_list



class Document:
    
    def __init__(self, reference_terms: BooleanOperation | str, abstract: str = "", title: str = "", doc_id: str = "1", 
                 weight: float = 1, ranking_position: int = 1):
        self.__reference_terms = reference_terms
        self.__abstract = abstract
        self.__title = title
        self.__doc_id = doc_id
        self.__weight = weight
        self.__ranking_position = ranking_position
        self.__sentences: list[Sentence] = []
        self.__graphs: list[Graph] = self.__create_and_get_graph_list()

        self.__calculate_sentences_list_from_documents()
    

    def get_abstract(self) -> str:
        return self.__abstract
    

    def get_title(self) -> str:
        return self.__title
    

    def get_doc_id(self) -> str:
        return self.__doc_id
    

    def get_reference_terms(self) -> BooleanOperation | str:
        return self.__reference_terms
    

    def get_sentences(self) -> list[Sentence]:
        return self.__sentences
    

    def get_weight(self) -> float:
        return self.__weight
    

    def get_ranking_position(self) -> int:
        return self.__ranking_position
    

    def get_graphs(self) -> list[Graph]:
        return self.__graphs


    def get_ieee_explore_article(self,
            parameter: str, 
            value: str
            ) -> None:
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
    

    def do_text_transformations_by_sentence(self,
            stop_words_list: list[str], 
            lema: bool,
            stem: bool
            ) -> None:
        """
        Apply some text transformations to each sentence of the document and calculate term positions to the sentence.

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
            sentence.do_text_transformations_and_term_positions_dict(stop_words_list, lema, stem)
    

    def set_graph_attributes_to_doc_and_sentences(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_ref_terms: bool = True, 
            format_adjac_refterms: bool  = True):
        #Set the attributes to the graphs of the document
        for graph in self.__graphs:
            graph.set_graph_attributes(nr_of_graph_terms, limit_distance, include_ref_terms, format_adjac_refterms)
        
        #Set the attributes to the graphs of each sentence of the document
        for sentence in self.__sentences:
            for graph in sentence.get_graphs():
                graph.set_graph_attributes(nr_of_graph_terms, limit_distance, include_ref_terms, format_adjac_refterms)
        
    
    def do_text_transformations_to_refterms(self,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> None:
        """
        Apply some text transformations to the document. Remove stopwords, punctuation, stemming, lematization.

        Parameters
        ----------
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        """
        if type(self.__reference_terms) == str:
            if len(self.__reference_terms) > 0:
                sentence_from_refterm = Sentence(raw_text=self.__reference_terms)
                self.__reference_terms = sentence_from_refterm.get_transformed_sentence_str(stop_words_list, lema, stem)
        elif type(self.__reference_terms) == BooleanOperation: 
            self.__reference_terms.do_text_transformations_to_operands(stop_words_list, lema, stem)


    def __calculate_sentences_list_from_documents(self) -> None:
        """
        Transform the text of each document in the input list into a list of sentences. Split the text by dots.
        """
        sentence_list = [self.__title]
        abstract_sentence_list = self.__abstract.split('. ')
        sentence_list.extend(abstract_sentence_list)
        for index, sentence_str in enumerate(sentence_list):
            sentence_obj = Sentence(raw_text=sentence_str, reference_terms=self.__reference_terms, position_in_doc=index)
            self.__sentences.append(sentence_obj)
    

    def __create_and_get_graph_list(self) -> list[Graph]:
        """
        Create a list of graphs associated with the document
        """
        graph_list = []
        if type(self.__reference_terms) == str:
            graph_list.append(Graph(reference_terms=self.__reference_terms))
        elif type(self.__reference_terms) == BooleanOperation: 
            for ref_term in self.__reference_terms.get_operands_str_list():
                graph_list.append(Graph(reference_terms=ref_term))
        
        return graph_list



class Ranking:
    
    def __init__(self, query_text: str, reference_terms: BooleanOperation | str, nr_search_results: int = 10, ranking_weight_type: str = 'linear', 
                 stop_words: list[str] = [], lema: bool = True, stem: str = False):
        self.__query_text = query_text
        self.__reference_terms = reference_terms
        self.__nr_search_results = nr_search_results
        self.__ranking_weight_type = ranking_weight_type   #Type of weighting to be applied (it can be: 'none', 'linear' or 'inverse')
        self.__stop_words = stop_words
        self.__lema = lema
        self.__stem = stem
        self.__documents: list[Document] = []

        self.__do_text_transformations_to_refterms()
        results = self.__get_ieee_explore_ranking()
        self.__calculate_ranking_as_weighted_documents_and_do_text_transformations(results)

        self.__graph = Graph(reference_terms=self.__reference_terms)


    def get_documents(self) -> list[Document]:
        return self.__documents 
    

    def get_graph(self) -> Graph:
        return self.__graph
    

    def get_reference_terms(self) -> BooleanOperation | str:
        return self.__reference_terms
    

    def set_graph_attributes_to_ranking_and_documents(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_ref_terms: bool = True, 
            format_adjac_refterms: bool = True
            ) -> None:
        """
        Set graph attributes to the graph associated with the current object and each Document object from the documents attribute.

        Parameters
        ----------
        nr_of_graph_terms : int
            Configured number of terms in the graph
        limit_distance : int
            Maximal distance of terms used to calculate the vecinity
        include_ref_terms : bool
            If True, the reference term is included in the vecinity
        format_adjacent_refterms : bool
            If True, format terms only if they are adjacent
        """
        self.__graph.set_graph_attributes(nr_of_graph_terms, limit_distance, include_ref_terms, format_adjac_refterms)

        for document in self.__documents:
            document.set_graph_attributes_to_doc_and_sentences(nr_of_graph_terms, limit_distance, include_ref_terms, format_adjac_refterms)


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


    def __do_text_transformations_to_refterms(self) -> None:
        """
        Apply text transformations to the reference terms of the ranking. Remove stopwords, punctuation, stemming, lematization.
        """
        if type(self.__reference_terms) == str:
            if len(self.__reference_terms) > 0:
                sentence_from_refterm = Sentence(raw_text=self.__reference_terms)
                self.__reference_terms = sentence_from_refterm.get_transformed_sentence_str(self.__stop_words, self.__lema, self.__stem)
        elif type(self.__reference_terms) == BooleanOperation: 
            self.__reference_terms.do_text_transformations_to_operands(self.__stop_words, self.__lema, self.__stem)


    def __calculate_ranking_as_weighted_documents_and_do_text_transformations(self,
            results: list[dict],
            ) -> None:
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
            new_doc = Document(reference_terms=self.__reference_terms, abstract=_abstract, title=_title, doc_id=_doc_id, 
                               weight=_weight, ranking_position=_ranking_pos)
            new_doc.do_text_transformations_by_sentence(self.__stop_words, self.__lema, self.__stem)
            self.__documents.append(new_doc)


    def __calculate_weight(self,
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



class Graph:
        
    def __init__(self, reference_terms: BooleanOperation | str, nr_of_graph_terms: int = 5, limit_distance: int = 4, include_ref_terms: bool = True, 
                 format_adjac_refterms: bool = True):

        #self.nodes = {}
        self.__reference_terms = reference_terms
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_reference_terms = include_ref_terms
        self.__format_adjacent_refterms = format_adjac_refterms

        self.__vecinity_matrix = []
        self.__most_freq_dist_list = []
        


    def set_graph_attributes(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_ref_terms: bool = True, 
            format_adjac_refterms: bool  = True
            ) -> None:
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_reference_terms = include_ref_terms
        self.__format_adjacent_refterms = format_adjac_refterms


    def get_reference_terms(self) -> BooleanOperation | str:
        return self.__reference_terms


    def get_number_of_graph_terms(self) -> int:
        return self.__number_of_graph_terms


    def get_limit_distance(self) -> int:
        return self.__limit_distance


    def get_include_reference_terms(self) -> bool:
        return self.__include_reference_terms


    def get_format_adjacent_refterms(self) -> bool:
        return self.__format_adjacent_refterms
    

    def get_vecinity_matrix(self) -> list[dict]:
        return self.__vecinity_matrix
    

    def get_most_freq_dist_list(self) -> list[dict]:
        return self.__most_freq_dist_list
