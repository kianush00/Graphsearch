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
                    sentence_from_operand = Sentence(raw_text=operand)
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
    
    def __init__(self, raw_text: str, reference_terms: BooleanOperation | str = "", position_in_doc: int = 0, weight: float = 1.0):
        self.__raw_text = raw_text
        self.__reference_terms = reference_terms
        self.__position_in_doc = position_in_doc
        self.__weight = weight
        self.__processed_text: str = ""
        self.__graphs: list[Graph] = self.__create_and_get_graph_list()
        self.__term_positions_dict: defaultdict[str, list[int]] = defaultdict(list)
        self.__ref_terms_positions_dict: dict[str, list[int]] = {}
        self.__vicinity_matrix: dict[str, dict[str, list[float]]] = []


    def get_raw_text(self) -> str:
        return self.__raw_text
    

    def get_reference_terms(self) -> BooleanOperation | str:
        return self.__reference_terms


    def get_position_in_doc(self) -> int:
        return self.__position_in_doc
    

    def get_weight(self) -> float:
        return self.__weight


    def get_processed_text(self) -> str:
        return self.__processed_text
    

    def set_processed_text(self, value: str) -> None:
        self.__processed_text = value
    

    def get_graphs(self) -> list[Graph]:
        return self.__graphs
    

    def get_vicinity_matrix(self) -> dict[str, dict[str, list[float]]]:
        return self.__vicinity_matrix


    def do_text_transformations_if_any_refterm(self,
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
    

    def calculate_term_positions_and_vicinity_matrix(self) -> None:
        """
        Calculate term positions dictionary for terms of the sentence and terms from
        the reference terms, also calculates the vicinity of a list of reference 
        terms in the current sentence, limited by a specified distance.
        """
        self.__do_term_positions_dict()
        self.__do_ref_term_positions_dict_conditioned_by_ref_type()
        self.__do_vicinity_matrix()
    

    def generate_graph_nodes_of_sentence(self) -> None:
        """
        Generate nodes to all the graphs associated with the sentence, based on 
        their isolated reference terms
        """
        for graph in self.__graphs:
            self.__generate_graph_nodes(graph)
    

    def __generate_graph_nodes(self, graph: Graph) -> None:
        """
        Generate nodes for the graph associated with the sentence, based on the isolated reference term of 
        the graph. This method calculates the ponderations of the terms associated with
        the reference term of the graph, and adds each of them to their respective node.

        Parameters
        ----------
        graph : Graph
            A graph associated with the sentence
        """
        reference_term = graph.__reference_terms
        terms_pond_dict = self.__get_terms_ponderation_dict(reference_term)

        # Iterate over each term in the frequency dictionary
        for neighbor_term in terms_pond_dict.keys():
            # Retrieve the list of frequencies by distance for the term
            list_of_freq_by_distance = self.__vicinity_matrix.get(neighbor_term).get(reference_term)
            distance_calculation_list = []
            # Construct a list of distances multiplied by its frequencies
            for idx, freq_mult_by_weight in enumerate(list_of_freq_by_distance):
                frequency = round(freq_mult_by_weight / self.__weight)
                distance_calculation_list.extend([idx+1] * frequency)
                
             # Calculate the mean distance for the term
            new_node = Node(term=neighbor_term, ponderation=terms_pond_dict.get(neighbor_term), distance=np.mean(distance_calculation_list))
            graph.add_node(new_node)

    

    def __get_terms_ponderation_dict(self, 
            reference_term: str
            ) -> dict[str, float]:
        """
        This method calculates the ponderation (frequency * weight) of terms from the 
        vecinity matrix of the sentence, by a specified reference term

        Parameters
        ----------
        reference_term : str
            The reference term of the graph

        Returns
        -------
        terms_pond_dict : dict[str, float]
            A dictionary containing the ponderation of terms. Each dictionary has keys corresponding 
            to terms, and values represent the ponderation of the term within the sentence.

        """
        terms_pond_dict = {}

        # Iterate over each term and its ponderations in the document
        for neighbor_term, distance_pond_by_ref_term in self.__vicinity_matrix.items():
            # Split the reference terms into a list of words
            ref_term_splitted = reference_term.split(" ")
            for ref_word in ref_term_splitted:
                # Checks if the reference word is in the distance-ponderation 
                # dictionary keys and if the neighbor term isn't a ref term
                if (ref_word in distance_pond_by_ref_term) and (neighbor_term not in ref_term_splitted):
                    sum_of_ponds_in_ref_term = sum(distance_pond_by_ref_term[ref_word])
                    if sum_of_ponds_in_ref_term > 0:
                        terms_pond_dict[neighbor_term] = sum_of_ponds_in_ref_term
            
        return terms_pond_dict


    def __do_vicinity_matrix(self) -> None:
        """
        Calculate the vicinity of a list of reference terms in the current sentence, limited by a specified distance.

        E.g. {'v_term_1': {'ref_term_1': [0.0, 1.8, 0.0, 0.9]}, 
        'v_term_2': {'ref_term_1': [0.0, 0.0, 0.9, 0.0], 'ref_term_2': [2.7, 0.0, 0.0, 0.9]}, ... }
        """
        vicinity_matrix = {}  # Create the empty dictionary
        # Calculate all terms in term_positions_defaultdict that are at distance limit_distance (or closer) to the reference_terms
        # and return a list of these terms and their corresponding distances
        for term, term_positions in self.__term_positions_dict.items():
            #Avoid comparing the ref term with itself (if bool false)
            if((term not in self.__ref_terms_positions_dict.keys()) or (self.__graphs[0].get_include_reference_terms())): 
                # Calculate the distance between the reference term and the rest of terms
                first_one = True
                for ref_term, ref_positions in self.__ref_terms_positions_dict.items():
                    if ref_term != term:
                        freq_neighborhood_positions = self.__calculate_ponderation_of_distances_between_term_positions(ref_positions, 
                                                                                                                       term_positions)

                        if (any(frq > 0 for frq in freq_neighborhood_positions)):
                            if (first_one):
                                vicinity_matrix[term] = {}
                                first_one = False
                            vicinity_matrix[term][ref_term] = freq_neighborhood_positions

        self.__vicinity_matrix = vicinity_matrix


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

    
    def __do_ref_term_positions_dict_conditioned_by_ref_type(self) -> None:
        """
        Returns a dictionary to store the list of positions for each reference term, along with its splitted terms.
        This function takes a defaultdict containing term positions and a list of reference terms as input.
        It splits each reference term into individual words and retrieves the positions of each term from the defaultdict.
        If a reference term contains multiple words, the function retrieves positions for each individual word and combines them.
        The resulting dictionary stores each reference term along with its list of positions.
        """
        ref_term_positions_dict = {}

        if type(self.__reference_terms) == str:
            if len(self.__reference_terms) > 0:
                ref_term = self.__reference_terms
                ref_term_positions_dict = self.__get_ref_term_positions_dict(ref_term)
        elif type(self.__reference_terms) == BooleanOperation: 
            for ref_term in self.__reference_terms.get_operands_str_list():
                ref_term_positions_dict.update(self.__get_ref_term_positions_dict(ref_term))
        
        self.__ref_terms_positions_dict = ref_term_positions_dict


    def __get_ref_term_positions_dict(self, ref_term: str) -> None:
        """
        Returns a dictionary to store the list of positions for each reference term, along with its splitted terms.
        This function takes a defaultdict containing term positions and a list of reference terms as input.
        It splits each reference term into individual words and retrieves the positions of each term from the defaultdict.
        If a reference term contains multiple words, the function retrieves positions for each individual word and combines them.
        The resulting dictionary stores each reference term along with its list of positions.
        """
        ref_term_positions_dict = {}

        ref_term_words = ref_term.split(' ')
        # If the reference term is within the term position dictionary
        if ref_term_words[0] in self.__term_positions_dict.keys():
            # If the reference term contains more than one word
            if (len(ref_term_words) > 1):
                for splitted_ref_term in ref_term_words:   
                    # Get the term positions of each splitted reference term
                    ref_term_positions_dict[splitted_ref_term] = self.__term_positions_dict[splitted_ref_term]
                if self.__graphs[0].get_format_adjacent_refterms():
                    ref_term_positions_dict.update(self.__format_adjacent_ref_term_positions_dict(ref_term_positions_dict, ref_term_words))
            # If the reference term contains just one word
            else:
                # Get the term positions of the reference term
                ref_term_positions_dict[ref_term] = self.__term_positions_dict[ref_term]
        
        return ref_term_positions_dict



    def __format_adjacent_ref_term_positions_dict(self,
            ref_term_positions_dict: dict[str, list[int]],
            reference_term_words: list[str]
            ) -> dict[str, list[int]]:
        """
        Format the reference term positions dictionary, so that it can be used in the vicinity matrix.
        This function takes a reference term positions dictionary and a list of reference term words as input.
        It creates a new dictionary containing only the keys present in the reference term words list,
        along with their corresponding lists of positions.

        Parameters
        ----------
        ref_term_positions_dict : dict[str, list[int]]
            A dictionary containing term positions.
        reference_term_words : list[str]
            A list of reference term words.

        Returns
        -------
        formatted_dict : dict[str, list[int]]
            A formatted dictionary with reference term words as keys and lists of positions as values.
        """
        ref_term_positions_dict_splitted = defaultdict(list)

        #Omit keys that are not in reference_term_words
        for key, value in ref_term_positions_dict.items():
            if key in reference_term_words:
                ref_term_positions_dict_splitted[key] = value
        
        new_dict = self.__get_new_dict_formatted_adjacent_ref_term_positions(reference_term_words, ref_term_positions_dict_splitted)
        formatted_dict = self.__get_new_dict_formatted_adjacent_ref_term_positions(reference_term_words, new_dict)
        
        return formatted_dict


    def __get_new_dict_formatted_adjacent_ref_term_positions(self,
            reference_terms: list[str],
            ref_term_positions_dict: dict[str, list[int]]
            ) -> dict[str, list[int]]:
        """
        Get a new formatted reference term positions dictionary.
        This function takes a list of reference terms and a dictionary containing term positions as input.
        It iterates over the reference terms and retrieves the positions of adjacent terms in the reference terms list.
        The resulting dictionary contains each reference term word as a key and its list of positions as the value.

        Parameters
        ----------
        reference_terms : list[str]
            A list of reference terms.
        ref_term_positions_dict : dict[str, list[int]]
            A dictionary containing term positions.

        Returns
        -------
        new_formatted_dict : dict[str, list[int]]
            A new formatted dictionary with reference term words as keys and lists of positions as values.
        """
        new_formatted_dict = defaultdict(list)

        for index in range(len(reference_terms)-1):
            for number1 in ref_term_positions_dict[reference_terms[index]]:
                for number2 in ref_term_positions_dict[reference_terms[index+1]]:
                    if ((number1 + 1) == number2):
                        if index == 0:
                            new_formatted_dict[reference_terms[index]].append(number1)
                        new_formatted_dict[reference_terms[index+1]].append(number2)
        
        return new_formatted_dict
    

    def __calculate_ponderation_of_distances_between_term_positions(self,
            term1_positions: list[int], 
            term2_positions: list[int]
            ) -> list[float]:
        """
        Compare the positions vectors of two terms, and return the ponderation 
        (frequency * weight) list per distance between reference terms and vicinity terms

        Parameters
        ----------
        term1_positions : list[int]
            List of positions of the first term
        term2_positions : list[int]
            List of positions of the second term

        Returns
        -------
        ponderations_per_distance : list[float]
            List of ponderations per distance between reference terms and vicinity terms
        """
        limit_distance = self.__graphs[0].get_limit_distance()
        ponderations_per_distance = [0] * limit_distance

        for term1_pos in term1_positions:
            for term2_pos in term2_positions:
                absolute_distance = abs(term1_pos-term2_pos)
                if (absolute_distance <= limit_distance):
                    ponderations_per_distance[absolute_distance-1] += 1
        
        ponderations_per_distance = [i * self.__weight for i in ponderations_per_distance]
        return ponderations_per_distance
    

    def __create_and_get_graph_list(self) -> list[Graph]:
        """
        Create a list of graphs associated with the sentence. Each reference 
        term per graph is a string value.
        """
        graph_list = []
        if type(self.__reference_terms) == str:
            graph_list.append(Graph(reference_terms=self.__reference_terms))
        elif type(self.__reference_terms) == BooleanOperation: 
            for ref_term in self.__reference_terms.get_operands_str_list():
                graph_list.append(Graph(reference_terms=ref_term))
        
        return graph_list
    

    def __sort_terms_pond_dictionary(self,
            dictionary: dict[str, float]
            ) -> dict[str, float]:
        """Function to sort the dictionary by values in descending order."""
        sorted_terms_pond_dict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
        return sorted_terms_pond_dict



class Document:
    
    def __init__(self, reference_terms: BooleanOperation | str, abstract: str = "", title: str = "", doc_id: str = "1", 
                 weight: float = 1.0, ranking_position: int = 1):
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
            sentence.do_text_transformations_if_any_refterm(stop_words_list, lema, stem)
    

    def set_graph_attributes_to_doc_and_sentences(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_refterms: bool = True, 
            format_adjac_refterms: bool  = True
            ) -> None:
        """
        Set graph attributes to the documents and sentences of the document

        Parameters
        ----------
        nr_of_graph_terms : int
            Configured number of terms in the graph
        limit_distance : int
            Maximal distance of terms used to calculate the vicinity
        include_reference_term : bool
            If True, the reference term is included in the vicinity
        format_adjacent_refterms : bool
            If True, format terms only if they are adjacent
        """
        #Set the attributes to the graphs of the document
        for graph in self.__graphs:
            graph.set_graph_attributes(nr_of_graph_terms, limit_distance, include_refterms, format_adjac_refterms)
        
        #Set the attributes to the graphs of each sentence of the document. Sentences have no graph term limit.
        for sentence in self.__sentences:
            for graph in sentence.get_graphs():
                graph.set_graph_attributes(nr_of_graph_terms=999999, limit_distance=limit_distance, 
                                           include_refterms=include_refterms, format_adjac_refterms=format_adjac_refterms)
        
    
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


    def generate_graph_nodes_of_doc_and_sentences(self) -> None:
        """
        Generate all the nodes associated with the document graphs, along with the graphs of their
        sentences, based on their reference terms
        """
        for sentence in self.__sentences:
            sentence.generate_graph_nodes_of_sentence()


    def __calculate_sentences_list_from_documents(self) -> None:
        """
        Transform the text of each document in the input list into a list of sentences. Split the text by dots.
        """
        sentence_list = [self.__title]
        abstract_sentence_list = self.__abstract.split('. ')
        sentence_list.extend(abstract_sentence_list)
        for index, sentence_str in enumerate(sentence_list):
            sentence_obj = Sentence(raw_text=sentence_str, reference_terms=self.__reference_terms, position_in_doc=index, weight=self.__weight)
            self.__sentences.append(sentence_obj)
    

    def __create_and_get_graph_list(self) -> list[Graph]:
        """
        Create a list of graphs associated with the sentence. Each reference 
        term per graph is a string value.
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
    

    def set_attributes_to_all_graphs_and_calculate_vicinity_matrix(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_refterms: bool = True, 
            format_adjac_refterms: bool = True
            ) -> None:
        """
        Set graph attributes to the graph associated with the current object and each Document object from the documents attribute.

        Parameters
        ----------
        nr_of_graph_terms : int
            Configured number of terms in the graph
        limit_distance : int
            Maximal distance of terms used to calculate the vicinity
        include_ref_terms : bool
            If True, the reference term is included in the vicinity
        format_adjacent_refterms : bool
            If True, format terms only if they are adjacent
        """
        self.__graph.set_graph_attributes(nr_of_graph_terms, limit_distance, include_refterms, format_adjac_refterms)

        for document in self.__documents:
            document.set_graph_attributes_to_doc_and_sentences(nr_of_graph_terms, limit_distance, include_refterms, format_adjac_refterms)

        for document in self.__documents:
            for sentence in document.get_sentences():
                sentence.calculate_term_positions_and_vicinity_matrix()
    

    def generate_all_node_graphs(self):
        """
        Generate all the nodes associated with the graphs from both the ranking and all the documents along with their 
        sentences, based on the reference terms
        """
        for document in self.__documents:
            document.generate_graph_nodes_of_doc_and_sentences()



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



class Node:
    
    def __init__(self, term: str, ponderation: float = 1.0, distance: float = 1.0):
        self.__term = term
        self.__ponderation = ponderation
        self.__distance = distance
    

    def get_term(self) -> str:
        return self.__term
    

    def get_ponderation(self) -> float:
        return self.__ponderation
    

    def get_distance(self) -> float:
        return self.__distance



class Graph:
        
    def __init__(self, reference_terms: BooleanOperation | str, nr_of_graph_terms: int = 5, limit_distance: int = 4, include_refterms: bool = True, 
                 format_adjac_refterms: bool = True):
        self.__reference_terms = reference_terms
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_reference_terms = include_refterms
        self.__format_adjacent_refterms = format_adjac_refterms
        self.__nodes: list[Node] = []


    def set_graph_attributes(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_refterms: bool = True, 
            format_adjac_refterms: bool  = True
            ) -> None:
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_reference_terms = include_refterms
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
    

    def get_nodes(self) -> list[Node]:
        return self.__nodes
    

    def add_node(self, node: Node) -> None:
        self.__nodes.append(node)
