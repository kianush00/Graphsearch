#import srex_new
import operator
import numpy as np
import math
from collections import defaultdict
#import pandas as pd
from nltk.stem import PorterStemmer   #Stemmer
from textblob import Word #Lemmatize
from graphviz import Graph
import nltk
from functools import reduce
from xploreapi import XPLORE
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from numpy import dot
from enum import Enum
import copy
from collections import deque 


class TextUtils:

    @staticmethod
    def get_transformed_text_if_it_has_underscores(
            text_with_underscores: str,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> str:
        """
        Apply some transformations to the text with underscores. First the underscores are replaced, then 
        the text is transformed, and finally the underscores are reinserted into the text

        Parameters
        ----------
        text_with_underscores : str
            Text with underscores to be transformed
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lemmatization is applied
        stem : bool
            If True, stemming is applied
        
        Returns
        -------
        transformed_text : str
            The transformed sentence
        """
        text_with_underscores = text_with_underscores.replace('_', ' ')
        transformed_text = TextUtils.get_transformed_text(text_with_underscores, stop_words_list, lema, stem)
        transformed_text = transformed_text.replace(' ', '_')
        
        return transformed_text

    
    @staticmethod
    def get_transformed_text(
            text: str,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> str:
        """
        Apply some transformations to the text. Remove stopwords, punctuation, stemming, lemmatization.

        Parameters
        ----------
        text : str
            Text to be transformed
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lemmatization is applied
        stem : bool
            If True, stemming is applied
        
        Returns
        -------
        transformed_text : str
            The transformed sentence
        """
        # Convert the string to lowercase
        sentence = text.lower()
        
        # Tokenize and remove punctuation
        tokens = nltk.word_tokenize(sentence)
        filtered_sentence = [token for token in tokens if token.isalnum()]
        
        # Remove stopwords
        if len(stop_words_list) > 0:
            filtered_sentence = [word for word in filtered_sentence if word not in stop_words_list]
        
        # Apply lemmatization
        if lema:
            filtered_sentence = [Word(word).lemmatize() for word in filtered_sentence]
        
        # Apply stemming
        if stem:
            st = PorterStemmer()
            filtered_sentence = [st.stem(word) for word in filtered_sentence]
        
        # Join the tokens back into a single string
        transformed_text = ' '.join(filtered_sentence)
        
        return transformed_text



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
                    self.__operands[index] = TextUtils.get_transformed_text(sentence_from_operand.get_raw_text(), 
                                                                            stop_words_list, lema, stem)
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



class VicinityNode:
    
    def __init__(self, term: str, ponderation: float = 1.0, distance: float = 1.0):
        self.__term = term
        self.__ponderation = ponderation
        self.__distance = distance
    

    def get_term(self) -> str:
        return self.__term
    

    def get_ponderation(self) -> float:
        return self.__ponderation
    

    def set_ponderation(self, ponderation: float) -> None:
        self.__ponderation = ponderation
    

    def get_distance(self) -> float:
        return self.__distance
    

    def set_distance(self, distance: float) -> None:
        self.__distance = distance



class VicinityGraph:
        
    def __init__(self, reference_terms: BooleanOperation | str, nr_of_graph_terms: int = 5, 
                 limit_distance: int = 4, include_refterms: bool = True):
        self.__reference_terms = reference_terms
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_reference_terms = include_refterms
        self.__nodes: list[VicinityNode] = []


    def get_reference_terms(self) -> BooleanOperation | str:
        return self.__reference_terms


    def get_number_of_graph_terms(self) -> int:
        return self.__number_of_graph_terms


    def get_limit_distance(self) -> int:
        return self.__limit_distance


    def get_include_reference_terms(self) -> bool:
        return self.__include_reference_terms
    

    def get_nodes(self) -> list[VicinityNode]:
        return self.__nodes
    

    def get_node_by_term(self, term: str) -> VicinityNode:
        for node in self.__nodes:
            if node.get_term() == term:
                return node
        return None
    

    def get_node_terms(self) -> list[str]:
        node_terms = []
        for node in self.__nodes:
            node_terms.append(node.get_term())
        return node_terms


    def add_node(self, node: VicinityNode) -> None:
        self.__nodes.append(node)
    

    def delete_node_by_term(self, term: str) -> None:
        for node in self.__nodes:
            if node.get_term() == term:
                self.__nodes.remove(node)
                break
    

    def get_union_to_graph(self,
            external_graph
            ):
        """
        Unites an external graph with the own graph and obtains a new graph
        The merging process involves iterating through the nodes of both graphs, calculating 
        the sum of weights and the average distances between each one.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be united

        Returns
        -------
        united_graph : VicinityGraph
            The union between copy of the graph itself and an external graph
        """
        united_graph = self.__get_calculation_of_intersected_terms(external_graph)
        
        node_terms_from_copy_graph = united_graph.get_node_terms()
        node_terms_from_ext_graph = external_graph.get_node_terms()
        #if the external graph has exclusive terms, then it needs to be added to the united graph
        for node_term in set(node_terms_from_ext_graph) - set(node_terms_from_copy_graph):
            node_from_ext_graph = external_graph.get_node_by_term(node_term)
            united_graph.add_node(node_from_ext_graph)

        return united_graph
    

    def get_intersection_to_graph(self,
            external_graph
            ):
        """
        Intersects an external graph with the own graph and obtains a new graph
        The merging process involves iterating through the nodes of both graphs, calculating 
        the sum of weights and the average distances between each one.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be intersected

        Returns
        -------
        intersected_graph : VicinityGraph
            The intersection between copy of the graph itself and an external graph
        """
        #if the graph copy term is already in the external graph
        intersected_graph = self.__get_calculation_of_intersected_terms(external_graph)

        node_terms_from_copy_graph = intersected_graph.get_node_terms()
        node_terms_from_ext_graph = external_graph.get_node_terms()
        #if the graph copy has exclusive terms, then it needs to be deleted
        for node_term in set(node_terms_from_copy_graph) - set(node_terms_from_ext_graph):
            intersected_graph.delete_node_by_term(node_term)

        return intersected_graph
    

    def __get_calculation_of_intersected_terms(self,
            external_graph
            ):
        """
        Calculates the sum of weights and the average distances of the nodes between 
        the external graph and itself.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to calculate the intersected terms

        Returns
        -------
        copy_graph : VicinityGraph
            The copied graph that contain the calculation of the intersected terms
        """
        copy_graph = copy.deepcopy(self)

        node_terms_from_copy_graph = copy_graph.get_node_terms()
        node_terms_from_ext_graph = external_graph.get_node_terms()

        for node_term in set(node_terms_from_copy_graph) & set(node_terms_from_ext_graph):
            node_from_copy_graph = copy_graph.get_node_by_term(node_term)
            node_from_ext_graph = external_graph.get_node_by_term(node_term)
            #then calculate the average distances between the two nodes and the sum of the ponderations
            distance = np.mean([node_from_copy_graph.get_distance(), node_from_ext_graph.get_distance()])
            ponderation = node_from_copy_graph.get_ponderation() + node_from_ext_graph.get_ponderation()
            node_from_copy_graph.set_distance(distance)
            node_from_copy_graph.set_ponderation(ponderation)
        
        return copy_graph



class Sentence:
    
    def __init__(self, raw_text: str, reference_terms: BooleanOperation | str = "", position_in_doc: int = 0, weight: float = 1.0):
        self.__raw_text = raw_text
        self.__reference_terms = reference_terms
        self.__position_in_doc = position_in_doc
        self.__weight = weight
        self.__processed_text: str = ""
        self.__graphs: list[VicinityGraph] = []
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
    

    def get_graphs(self) -> list[VicinityGraph]:
        return self.__graphs
    

    def get_graph_by_reference_term(self, reference_term: str) -> VicinityGraph:
        for graph in self.__graphs:
            if graph.get_reference_terms() == reference_term:
                return graph
        return None
    

    def add_graph(self, graph: VicinityGraph) -> None:
        self.__graphs.append(graph)
    

    def get_vicinity_matrix(self) -> dict[str, dict[str, list[float]]]:
        return self.__vicinity_matrix


    def do_text_transformations_if_any_refterm(self,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> None:
        """
        Apply some text transformations to the sentence and calculate term positions to the 
        sentence. Also removes spaces from the reference terms.

        Parameters
        ----------
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        """
        transformed_sentence_str = TextUtils.get_transformed_text(self.__raw_text, stop_words_list, lema, stem)
        if type(self.__reference_terms) == str:
            if len(self.__reference_terms) > 0:
                if self.__reference_terms in transformed_sentence_str:
                    transformed_sentence_str = self.__get_transformed_sentence_str_without_spaces_in_refterms([self.__reference_terms], 
                                                                                                        transformed_sentence_str)
                    self.__processed_text = transformed_sentence_str
        elif type(self.__reference_terms) == BooleanOperation: 
            operands_str_list = self.__reference_terms.get_operands_str_list()
            #If there is any refterm in the transformed sentence string
            if any(ref_term in transformed_sentence_str for ref_term in operands_str_list):  
                transformed_sentence_str = self.__get_transformed_sentence_str_without_spaces_in_refterms(operands_str_list, 
                                                                                                        transformed_sentence_str)
                self.__processed_text = transformed_sentence_str
    

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
    

    def __get_transformed_sentence_str_without_spaces_in_refterms(
            self,
            operands_str_list: list[str],
            transformed_sentence_str: str) -> str:
        """
        Get the transformed sentence string without spaces in the reference terms

        Parameters
        ----------
        operands_str_list : list[str]
            List of operands from the reference terms
        transformed_sentence_str : str
            Transformed sentence string to remove spaces

        Returns
        -------
        sentence_str_without_spaces_in_refterms : str
            Transformed sentence string without spaces in reference terms within the sentence
        """
        sentence_str_without_spaces_in_refterms = transformed_sentence_str

        for ref_term in operands_str_list:
            #If the refterm is in the transformed sentence string and the refterm contains more than a word
            if (ref_term in transformed_sentence_str) and (len(ref_term.split(" ")) > 1): 
                ref_term_without_spaces = self.__remove_spaces(ref_term)
                sentence_str_without_spaces_in_refterms = sentence_str_without_spaces_in_refterms.replace(ref_term, 
                                                                                                          ref_term_without_spaces)
        return sentence_str_without_spaces_in_refterms
    

    def __generate_graph_nodes(self, 
            graph: VicinityGraph
            ) -> None:
        """
        Generate nodes for the graph associated with the sentence, based on the isolated reference term of 
        the graph. This method calculates the ponderations of the terms associated with
        the reference term of the graph, and adds each of them to their respective node.

        Parameters
        ----------
        graph : VicinityGraph
            A graph associated with the sentence
        """
        #the refterms of the sentence' graph are only string type, they are not boolean operations
        reference_term = graph.get_reference_terms() 
        # Remove spaces from the reference term
        reference_term = self.__remove_spaces(reference_term) 
        terms_pond_dict = self.__get_terms_ponderation_dict(reference_term)

        # Iterate over each term in the frequency dictionary
        for neighbor_term in terms_pond_dict.keys():
            # Retrieve the list of frequencies by distance for the term
            list_of_pond_by_distance = self.__vicinity_matrix.get(neighbor_term).get(reference_term)
            distance_calculation_list = []
            # Construct a list of distances multiplied by its frequencies
            for idx, freq_mult_by_weight in enumerate(list_of_pond_by_distance):
                frequency = round(freq_mult_by_weight / self.__weight)
                distance_calculation_list.extend([idx+1] * frequency)
                
             # Calculate the mean distance for the term
            new_node = VicinityNode(term=neighbor_term, ponderation=terms_pond_dict.get(neighbor_term), distance=np.mean(distance_calculation_list))
            graph.add_node(new_node)

    

    def __get_terms_ponderation_dict(self, 
            reference_term: str
            ) -> dict[str, float]:
        """
        This method calculates the ponderation (frequency * weight) of terms from the 
        vicinity matrix of the sentence, by a specified reference term

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
            # Checks if the reference word is in the distance-ponderation 
            # dictionary keys and if the neighbor term isn't a ref term
            if (reference_term in distance_pond_by_ref_term):
                sum_of_ponds_in_ref_term = sum(distance_pond_by_ref_term[reference_term])
                if sum_of_ponds_in_ref_term > 0:
                    terms_pond_dict[neighbor_term] = sum_of_ponds_in_ref_term
            
        return terms_pond_dict


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


    def __get_ref_term_positions_dict(self,
            ref_term: str
            ) -> dict[str, list[int]]:
        """
        Returns a dictionary to store the list of positions for each reference term, along with its splitted terms.
        This function takes a defaultdict containing term positions and a list of reference terms as input.
        It splits each reference term into individual words and retrieves the positions of each term from the defaultdict.
        The resulting dictionary stores each reference term along with its list of positions.
        """
        ref_term_positions_dict = {}

        # Remove refterm spaces, if any
        ref_term = self.__remove_spaces(ref_term)
        # If the reference term is within the term position dictionary
        if ref_term in self.__term_positions_dict.keys():
            # Get the term positions of the reference term
            ref_term_positions_dict[ref_term] = self.__term_positions_dict[ref_term]
        
        return ref_term_positions_dict
    

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
                # Iterate refterms that do not contain spaces
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
    

    def __remove_spaces(self, term: str) -> str:
        if (len(term.split(" ")) > 1):
            term = term.replace(" ", "")
        return term



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
        self.__graphs: list[VicinityGraph] = []

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
    

    def get_graphs(self) -> list[VicinityGraph]:
        return self.__graphs
    

    def get_graph_by_reference_term(self, reference_term: str) -> VicinityGraph:
        for graph in self.__graphs:
            if graph.get_reference_terms() == reference_term:
                return graph
        return None
    

    def add_graph(self, graph: VicinityGraph) -> None:
        self.__graphs.append(graph)
    

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


    def generate_graph_nodes_of_doc_and_sentences(self) -> None:
        """
        Generate all the nodes associated with the document graphs, along with the graphs of their
        sentences, based on their reference terms
        """
        #Generate graph nodes of sentences
        for sentence in self.__sentences:
            sentence.generate_graph_nodes_of_sentence()
        
        #Generate graph nodes of the current document
        for ref_term in self.__reference_terms.get_operands_str_list():
            graphs_from_sentences_with_refterm = self.__get_graphs_from_sentences_with_reference_term(ref_term)
            document_graph = self.__get_union_of_graphs(graphs_from_sentences_with_refterm)
            self.add_graph(document_graph)
    

    def initialize_sentence_graphs(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_refterms: bool = True
            ) -> None:
        """
        Initialize graphs associated to each sentence of the document. Sentences 
        have no graph term limit.

        Parameters
        ----------
        nr_of_graph_terms : int
            Configured number of terms in the graph
        limit_distance : int
            Maximal distance of terms used to calculate the vicinity
        include_reference_term : bool
            If True, the reference term is included in the vicinity
        """
        for sentence in self.__sentences:
            if type(self.__reference_terms) == str:
                new_graph = VicinityGraph(self.__reference_terms, nr_of_graph_terms, limit_distance, include_refterms)
                sentence.add_graph(new_graph)
            elif type(self.__reference_terms) == BooleanOperation: 
                for ref_term in self.__reference_terms.get_operands_str_list():
                    new_graph = VicinityGraph(ref_term, nr_of_graph_terms, limit_distance, include_refterms)
                    sentence.add_graph(new_graph)


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
                self.__reference_terms = TextUtils.get_transformed_text(sentence_from_refterm.get_raw_text(), 
                                                                        stop_words_list, lema, stem)
        elif type(self.__reference_terms) == BooleanOperation: 
            self.__reference_terms.do_text_transformations_to_operands(stop_words_list, lema, stem)
    

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


    def __calculate_sentences_list_from_documents(self) -> None:
        """
        Transform the text of each document in the input list into a list of sentences. Split the text by dots.
        """
        list_of_sentence_str = self.__get_list_of_sentence_strings()
        for index, sentence_str in enumerate(list_of_sentence_str):
            sentence_obj = Sentence(raw_text=sentence_str, reference_terms=self.__reference_terms, 
                                    position_in_doc=index, weight=self.__weight)
            self.__sentences.append(sentence_obj)

    
    def __get_list_of_sentence_strings(self) -> list[str]:
        """
        Transform the text of the current document (title + abstract) into a list of string 
        sentences. Split the text by dots.

        Returns
        -------
        list_of_sentence_str : list[str]
            List of sentence strings
        """
        list_of_sentence_str = [self.__title]
        abstract_list_of_sentence_str = self.__abstract.split('. ')
        list_of_sentence_str.extend(abstract_list_of_sentence_str)
        return list_of_sentence_str
    

    def __get_graphs_from_sentences_with_reference_term(self, 
            reference_term: str
            ) -> list[VicinityGraph]:
        """
        Get all the graphs from the sentences of the current document that have the indicated reference term.

        Parameters
        ----------
        reference_term : str
            Reference term to compare the graphs from the sentences

        Returns
        -------
        graphs_from_sentences_with_refterm : list[VicinityGraph]
            List of graphs from sentences with the indicated reference term
        """
        graphs_from_sentences_with_refterm = []
        for sentence in self.__sentences:
            graph_by_refterm = sentence.get_graph_by_reference_term(reference_term)
            graphs_from_sentences_with_refterm.append(graph_by_refterm)
        return graphs_from_sentences_with_refterm
    

    def __get_union_of_graphs(
            self, 
            graphs: list[VicinityGraph]
            ) -> VicinityGraph:
        """
        Get the union between the indicated graphs.

        Parameters
        ----------
        graphs : list[VicinityGraph]
            List of graphs to be united.

        Returns
        -------
        union_of_graphs : Graph
            The union between the graphs.
        """
        # reduce() applies a function of two arguments cumulatively to the items of a sequence or iterable, from left to right, so 
        # as to reduce the iterable to a single value. For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5)
        union_of_graphs = reduce((lambda graph1, graph2: graph1.get_union_to_graph(graph2)), graphs)
        return union_of_graphs



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

        self.__graph = VicinityGraph(reference_terms=self.__reference_terms)


    def get_documents(self) -> list[Document]:
        return self.__documents 
    

    def get_graph(self) -> VicinityGraph:
        return self.__graph
    

    def get_reference_terms(self) -> BooleanOperation | str:
        return self.__reference_terms
    

    def generate_all_graphs(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_refterms: bool = True
            ) -> None:
        """
        Initialize graphs associated to all the sentences of each document from the ranking. Sentences 
        have no graph term limit. Also calculates the vicinity matrix for each sentence. Finally,
        generates nodes of all graphs.

        Parameters
        ----------
        nr_of_graph_terms : int
            Configured number of terms in the graph
        limit_distance : int
            Maximal distance of terms used to calculate the vicinity
        include_ref_terms : bool
            If True, the reference term is included in the vicinity
        """
        #Set graph attributes to the graphs of each document and the sentences' graphs of each document
        self.__initialize_sentence_graphs(nr_of_graph_terms, limit_distance, include_refterms)

        #Calculate term positions and vicinity matrix of each sentence by document
        for document in self.__documents:
            for sentence in document.get_sentences():
                sentence.calculate_term_positions_and_vicinity_matrix()
        
        self.__generate_nodes_of_all_graphs()
    

    def __generate_nodes_of_all_graphs(self) -> None:
        """
        Generate all the nodes associated with the graphs from both the ranking and all the documents 
        along with their sentences, based on the reference terms.
        """
        for document in self.__documents:
            document.generate_graph_nodes_of_doc_and_sentences()
    

    def __initialize_sentence_graphs(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_refterms: bool = True
            ) -> None:
        """
        Initialize graphs associated to each sentence of each document from the ranking. Sentences 
        have no graph term limit.

        Parameters
        ----------
        nr_of_graph_terms : int
            Configured number of terms in the graph
        limit_distance : int
            Maximal distance of terms used to calculate the vicinity
        include_reference_term : bool
            If True, the reference term is included in the vicinity
        """
        for document in self.__documents:
            document.initialize_sentence_graphs(nr_of_graph_terms, limit_distance, include_refterms)


    def __do_text_transformations_to_refterms(self) -> None:
        """
        Apply text transformations to the reference terms of the ranking. Remove stopwords, punctuation, stemming, lematization.
        """
        if type(self.__reference_terms) == str:
            if len(self.__reference_terms) > 0:
                sentence_from_refterm = Sentence(raw_text=self.__reference_terms)
                self.__reference_terms = TextUtils.get_transformed_text(sentence_from_refterm.get_raw_text(), self.__stop_words, 
                                                                        self.__lema, self.__stem)
        elif type(self.__reference_terms) == BooleanOperation: 
            self.__reference_terms.do_text_transformations_to_operands(self.__stop_words, self.__lema, self.__stem)


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
        for index, article in enumerate(results):
            new_doc = self.__get_new_document_by_article(article, index, results_size=len(results))
            new_doc.do_text_transformations_by_sentence(self.__stop_words, self.__lema, self.__stem)
            self.__documents.append(new_doc)
    

    def __get_new_document_by_article(self, 
            article : dict, 
            index : int,
            results_size : int,
            ) -> Document:
        """
        Create and get a new Document object from an article.

        Parameters
        ----------
        article : dict
            Article obtained by the ranking from IEEE Xplore API
        index : int
            Index (position) of the document in the ranking
        results_size : int
            Number of ranking documents

        Returns
        -------
        new_doc : Document
            A new document obtained from the attributes of the article
        """
        # Calculate the weight depending on the argument value and the position of the document in the ranking
        _weight = self.__calculate_weight(self.__ranking_weight_type, results_size, index)
        _abstract = article.get('abstract', "")
        _title = article.get('title', "")
        _doc_id = article.get('article_number', "1")
        _ranking_pos = article.get('rank', 1)
        new_doc = Document(reference_terms=self.__reference_terms, abstract=_abstract, title=_title, doc_id=_doc_id, 
                            weight=_weight, ranking_position=_ranking_pos)
        
        return new_doc


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
