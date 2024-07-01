#import srex_new
#import operator
import numpy as np
import math
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer  
from graphviz import Graph
from functools import reduce
from xploreapi import XPLORE
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from numpy import dot
import copy
from collections import deque 
import re



class VectorUtils:

    @staticmethod
    def normalize_vector(
            vector: list[float], 
            new_min: float = 0, 
            new_max: float = 1.0
            ) -> list[float]:
        """
        Normalize the given vector to the new range [new_min, new_max] and return the normalized vector.

        Parameters
        ----------
        vector : list[float]
            Vector to normalize
        new_min : float, optional
            Minimum value of the new range to normalize
        new_max : float, optional
            Maximum value of the new range to normalize
        
        Returns
        -------
        normalized_vector : list[float]
            The normalized vector
        """
        # Get the previous vector range
        old_min = min(vector)
        old_max = max(vector)
        
        # Avoid division by zero if the vector has all the same values
        if old_max == old_min:
            return [new_min for _ in vector]
        
        # Normalize the vector to the new range using the formula:
        normalized_vector = [
            new_min + (new_max - new_min) * (value - old_min) / (old_max - old_min)
            for value in vector
        ]
        
        return normalized_vector
    
    
    @staticmethod
    def get_cosine_between_vectors(
            vector1: list[float], 
            vector2: list[float]
            ) -> float:
        """
        Calculate and return the cosine of the angle between the two given vectors

        Parameters
        ----------
        vector1 : list[float]
            First vector to calculate
        vector2 : list[float]
            Second vector to calculate
        
        Returns
        -------
        cosine_of_angle : float
            The cosine of the angle between the two vectors
        """
        # Avoid division by zero if the norm of any vector equals zero
        if norm(vector1) > 0 and norm(vector2) > 0:
            cosine_of_angle = dot(vector1, vector2) / norm(vector1) / norm(vector2)
        else:
            cosine_of_angle = 0
        
        return cosine_of_angle



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



class VicinityGraphConfig:
    def __init__(self, nr_of_graph_terms: int = 5, limit_distance: int = 4, 
                 include_query_terms: bool = True, summarize: str = 'mean'):
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_query_terms = include_query_terms
        self.__summarize = summarize    # it can be: 'mean' or 'median'
    

    def get_number_of_graph_terms(self) -> int:
        return self.__number_of_graph_terms


    def get_limit_distance(self) -> int:
        return self.__limit_distance


    def get_include_query_terms(self) -> bool:
        return self.__include_query_terms
    

    def get_summarize(self) -> str:
        return self.__summarize
    

    def get_config_params(self) -> tuple[int, int, bool, str]:
        return self.__number_of_graph_terms, self.__limit_distance, self.__include_query_terms, self.__summarize



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

    
    def __str__(self) -> str:
        term = self.__term
        ponderation = round(self.__ponderation, 1)
        distance = round(self.__distance, 1)
        string = f"TERM: {term} ; PONDERATION: {ponderation} ; DISTANCE: {distance}"
        return string



class VicinityGraph:
        
    def __init__(self, subquery: str, nr_of_graph_terms: int = 5, limit_distance: int = 4, 
                 include_query_terms: bool = True, summarize: str = 'mean'):
        self.subquery = subquery
        self.__config: VicinityGraphConfig = VicinityGraphConfig(nr_of_graph_terms, limit_distance, 
                                                                 include_query_terms, summarize)
        self.__nodes: list[VicinityNode] = []


    def get_config(self) -> VicinityGraphConfig:
        return self.__config
    

    def get_sorted_nodes_optionally_limited(self, limit: int = -1) -> list[VicinityNode]:
        """
        Return the list of nodes sorted by ponderation in descending order, and limit 
        the number of nodes in the list obtained.
        """
        # Sort the nodes in descending order
        sorted_nodes = sorted(self.__nodes, key=lambda node: node.get_ponderation(), reverse=True)
        # Limit the number of nodes, if specified
        if limit > 0: sorted_nodes = sorted_nodes[:limit]
        return sorted_nodes
    

    def get_node_by_term(self, term: str) -> VicinityNode | None:
        for node in self.__nodes:
            if node.get_term() == term:
                return node
        print("No node with term")
        return None
    

    def get_terms_from_nodes(self) -> list[str]:
        node_terms = []
        for node in self.get_sorted_nodes_optionally_limited():
            node_terms.append(node.get_term())
        return node_terms


    def add_node(self, node: VicinityNode) -> None:
        self.__nodes.append(node)


    def delete_node_by_term(self, term: str) -> None:
        for node in self.__nodes:
            if node.get_term() == term:
                self.__nodes.remove(node)
                return
        print("No node with term")
    

    def __str__(self) -> str:
        string = "SUBQUERY: " + self.subquery
        for node in self.get_sorted_nodes_optionally_limited():
            string += "\n" + str(node)
        return string
    

    def get_graph_as_dict(self) -> dict[str, dict[str, float]]:
        """
        Returns the graph as a dictionary.

        e.g.   {'v_term1': {'ponderation': 3.0, 'distance': 3.4}, 
                'v_term2': {'ponderation': 2.0, 'distance': 1.6}, ...}
        """
        graph_dict = {n.get_term(): {'ponderation': n.get_ponderation(), 
                                     'distance': n.get_distance()} for n in self.__nodes}
        return graph_dict

    
    def get_viewable_graph_copy(self) -> 'VicinityGraph':
        """
        Returns a copy of the graph, with the vicinity terms with the highest ponderation 
        limited by the number of graph terms configured in the current graph, and 
        ignores the vicinity nodes that include the query terms.
        """
        graph_copy = VicinityGraph(self.subquery, *self.__config.get_config_params())
        for node in self.__get_sorted_nodes_to_visualize():
            graph_copy.add_node(node)
        return graph_copy
    
    
    def get_graph_viz(self,
            node_size: str = '2', 
            node_color: str = 'green'
            ) -> Graph:
        """
        Visualizes the graph, highlighting the vicinity terms with the highest ponderation.

        Parameters
        ----------
        node_size : str, optional
            Node size to adjust the graph
        node_color : str, optional
            Node color to adjust the graph

        Returns
        -------
        visual_graph : Graph
            The graph to visualize its nodes
        
        """
        # The visual characteristics of the graph are configured
        visual_graph = Graph('G', filename='graph_search.gv', engine='neato')
        visual_graph.attr('node', shape='circle', fontsize='10')
        visual_graph.node('0', label=self.subquery, root='true', fixedsize='true', width=node_size, 
                          style='filled', fillcolor='azure3', fontcolor='black')
        
        viewable_graph = self.get_viewable_graph_copy()
        for index, node in enumerate(viewable_graph.get_sorted_nodes_optionally_limited()):
            node_distance = round(node.get_distance(), 1)
            p_with = str(node.get_ponderation())
            visual_graph.node("'" +str(index+1)+"'", node.get_term(), fixedsize='true', width=node_size, 
                              penwidth=p_with, color=node_color)
            visual_graph.edge('0', "'" +str(index+1)+"'", label=str(node_distance), len=str(node_distance))
            
        return visual_graph
    

    def get_cosine_similarity(self,
            external_graph: 'VicinityGraph',
            include_ponderation: bool = False
            ) -> float:
        """
        Calculate the cosine similarity with other graph. To calculate it, it is proposed 
        to compare the graph using a multidimensional vector space, where the each term 
        properties define a dimension of the space.

        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be compared
        include_ponderation : bool, optional
            Select whether to include ponderation of the nodes as a parameter to the comparison function

        Returns
        -------
        cosine_of_angle : float
            The cosine similarity between the current graph and the external graph
        """
        # Calculate the base vector with terms from the union between both vectors
        vector_base = self.__get_terms_from_union_between_graphs(external_graph)
        
        # Get the dictionaries of normalized nodes from each graph 
        normalized_self_nodes = self.get_normalized_nodes_dict()
        normalized_external_nodes = external_graph.get_normalized_nodes_dict()

        # Initialize the vectors
        self_vector = [] 
        external_vector = []

        # Calculate the two vectors in the multidimensional space, i.e. generate the vector space
        for term in vector_base: 
            if term in normalized_self_nodes.keys():
                self_vector.append(normalized_self_nodes[term]['distance'])
            else:
                self_vector.append(0)  # Distance value equal to zero

            if term in normalized_external_nodes.keys():
                external_vector.append(normalized_external_nodes[term]['distance'])
            else:
                external_vector.append(0)  # Distance value equal to zero
        
        # Add ponderation values to the two vectors if include ponderation is True
        if include_ponderation:
            for term in vector_base: 
                if term in normalized_self_nodes.keys():
                    self_vector.append(normalized_self_nodes[term]['ponderation'])
                else:
                    self_vector.append(0)  # Ponderation value equal to zero

                if term in normalized_external_nodes.keys():
                    external_vector.append(normalized_external_nodes[term]['ponderation'])
                else:
                    external_vector.append(0)  # Ponderation value equal to zero

        # Calculate the cosine of the angle between the vectors
        cosine_of_angle = VectorUtils.get_cosine_between_vectors(self_vector, external_vector)
        
        return cosine_of_angle
    

    def get_normalized_nodes_dict(self) -> dict[str, dict[str, float]]:
        """
        Return a dictionary with the normalized ponderations and distances of each node in the graph.
        Ponderation and distance values are between 0.5 and 1.

        Returns
        -------
        normalized_self_nodes : dict[str, dict[str, float]]
            A dictionary with normalized ponderations and distances of each node in the graph
            e.g.   {'v_term1': {'ponderation': 0.75, 'distance': 0.57}, 
                    'v_term2': {'ponderation': 0.63, 'distance': 0.82}, ...}
        """
        # Get the ponderations and distances from the graph 
        normalized_self_nodes = self.get_graph_as_dict()

        # Get the distance and ponderation vectors from the graph
        ponderation_vector = [value['ponderation'] for value in normalized_self_nodes.values()]
        distance_vector = [value['distance'] for value in normalized_self_nodes.values()]

        # Get the normalized version of the distance and ponderation vectors, in a range of [0.5, 1.0]
        normalized_ponderation_vector = VectorUtils.normalize_vector(ponderation_vector, 0.5, 1.0)
        normalized_distance_vector = VectorUtils.normalize_vector(distance_vector, 0.5, 1.0)

        # Assign the normalized values ​​to the dictionary
        for index, subdict in enumerate(normalized_self_nodes.values()):
            subdict['ponderation'] = normalized_ponderation_vector[index]
            subdict['distance'] = normalized_distance_vector[index]

        return normalized_self_nodes
    

    def get_union_to_graph(self,
            external_graph: 'VicinityGraph'
            ) -> 'VicinityGraph':
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
        
        node_terms_from_copy_graph = united_graph.get_terms_from_nodes()
        node_terms_from_ext_graph = external_graph.get_terms_from_nodes()

        #if the external graph has exclusive terms, then it needs to be added to the united graph
        for node_term in set(node_terms_from_ext_graph) - set(node_terms_from_copy_graph):
            node_from_ext_graph = external_graph.get_node_by_term(node_term)
            united_graph.add_node(node_from_ext_graph)

        return united_graph
    

    def get_intersection_to_graph(self,
            external_graph: 'VicinityGraph'
            ) -> 'VicinityGraph':
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

        node_terms_from_copy_graph = intersected_graph.get_terms_from_nodes()
        node_terms_from_ext_graph = external_graph.get_terms_from_nodes()
        #if the graph copy has exclusive terms, then it needs to be deleted
        for node_term in set(node_terms_from_copy_graph) - set(node_terms_from_ext_graph):
            intersected_graph.delete_node_by_term(node_term)

        return intersected_graph
    

    def __get_calculation_of_intersected_terms(self,
            external_graph: 'VicinityGraph'
            ) -> 'VicinityGraph':
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

        for node_term in copy_graph.__get_terms_from_intersection_between_graphs(external_graph):
            node_from_copy_graph = copy_graph.get_node_by_term(node_term)
            node_from_ext_graph = external_graph.get_node_by_term(node_term)

            #then calculate the average distances between the two nodes and the sum of the ponderations
            copy_distance = node_from_copy_graph.get_distance()
            copy_pond = node_from_copy_graph.get_ponderation()
            ext_distance = node_from_ext_graph.get_distance()
            ext_pond = node_from_ext_graph.get_ponderation()

            sum_of_ponds = copy_pond + ext_pond
            average_distance = ((copy_distance * copy_pond) + (ext_distance * ext_pond)) / sum_of_ponds

            #set new distance and ponderation to each intersected term
            node_from_copy_graph.set_distance(average_distance)
            node_from_copy_graph.set_ponderation(sum_of_ponds)
        
        return copy_graph
    

    def __get_terms_from_intersection_between_graphs(self,
            graph_b: 'VicinityGraph'
            ) -> set[str]:
        """
        Return the set of terms from the intersection between the current graph and an external graph
        """
        base_terms = set(self.get_terms_from_nodes()) & set(graph_b.get_terms_from_nodes())
        return base_terms
    

    def __get_terms_from_union_between_graphs(self,
            graph_b: 'VicinityGraph'
            ) -> set[str]:
        """
        Return the set of terms from the union between the current graph and an external graph
        """
        base_terms = set(self.get_terms_from_nodes()) | set(graph_b.get_terms_from_nodes())
        return base_terms
    

    def __get_sorted_nodes_to_visualize(self) -> list[VicinityNode]:
        """
        Return the list of nodes sorted by ponderation in descending order, and limit 
        its lenght by the number of graph terms, from the current graph configuration.
        It also ignores the vicinity nodes that include the query terms.
        """
        sorted_nodes_to_visualize = self.get_sorted_nodes_optionally_limited()
        subquery = self.subquery

        # Removes characters '(', ')' and spaces in subquery variable
        subquery = subquery.translate(str.maketrans("", "", "() "))

        # Gets a list from subquery splitted by any 'AND' or 'OR' character
        splitted_subquery = re.split(r'AND|OR', subquery)

        # Ignores the nodes that contain a subquery in its term
        sorted_nodes_to_visualize = [
            node for node in sorted_nodes_to_visualize if node.get_term() not in splitted_subquery
        ]

        # Limits the number of nodes to the number of graph terms
        sorted_nodes_to_visualize = sorted_nodes_to_visualize[:self.__config.get_number_of_graph_terms()]

        return sorted_nodes_to_visualize



class BinaryTreeNode:
    def __init__(self, value: str):
        self.value = value
        self.graph: VicinityGraph | None = None
        self.left: BinaryTreeNode | None = None
        self.right: BinaryTreeNode | None = None
    

    def is_leaf(self) -> bool:
        is_leaf = not self.left and not self.right
        return is_leaf
    

    def get_values_from_leaves(self) -> list[str]:
        if self.is_leaf():
            return [self.value]
        return self.left.get_values_from_leaves() + self.right.get_values_from_leaves()
    

    def get_leaves(self) -> list['BinaryTreeNode']:
        if self.is_leaf():
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()
    

    def get_graph_from_subtree_by_subquery(self, query: str) -> VicinityGraph | None:
        if self.graph and self.graph.subquery == query:
            return self.graph
            
        # Check left subtree
        if self.left:
            left_candidate = self.left.get_graph_from_subtree_by_subquery(query)
            if left_candidate:
                return left_candidate

        # Check right subtree
        if self.right:
            right_candidate = self.right.get_graph_from_subtree_by_subquery(query)
            if right_candidate:
                return right_candidate

        return None
    

    def get_union_to_subtree(self, 
            external_subtree: 'BinaryTreeNode'
            ) -> 'BinaryTreeNode':
        """
        Gets the union between an external subtree and the own subtree, then obtains 
        a new subtree. The merging process involves iterating through the nodes of both 
        subtrees, calculating  the sum of weights and the average distances between 
        each one, that is, the union between each graph (tree node) of both subtrees.
        
        Parameters
        ----------
        external_subtree : BinaryTreeNode
            The external subtree to be united

        Returns
        -------
        copy_subtree : BinaryTreeNode
            The union between copy of the subtree itself and an external subtree
        """
        copy_subtree = copy.deepcopy(self)
        copy_subtree.__do_union_between_copy_self_and_subtree(external_subtree)
        return copy_subtree
    

    def do_graph_operation_from_subtrees(self) -> None:
        if not self.is_leaf():
            #Validate if the tree has no empty vicinity graphs in leaves
            try:
                self.__set_and_get_graph_operated_from_subtrees()
            except Exception as e:
                print('Error at do_graph_operation_from_subtrees(): ' + repr(e))
    

    def __str__(self) -> str:
        return ''.join(self.__inorder_traversal())
    

    def tree_str(self) -> str:
        return self.__generate_tree_str()
    

    def __do_union_between_copy_self_and_subtree(self, 
            external_peer_node: 'BinaryTreeNode'
            ) -> None:
        """
        Unites an external subtree with the own copy subtree and modifies 
        the own subtree graphs. This method should only be used by the 
        copy from the original BinaryTreeNode object.
        
        Parameters
        ----------
        external_peer_node : BinaryTreeNode
            The external peer node to be united
        """
        #Checks if both nodes have a non-null graph attribute, to then join them and set 
        #them in the copy attribute
        if self.graph and external_peer_node.graph:
            current_node_united_graph = self.graph.get_union_to_graph(external_peer_node.graph)
            self.graph = current_node_united_graph
        
        #Checks if both nodes have a left node, to continue the recursion
        if self.left and external_peer_node.left:
            self.left.__do_union_between_copy_self_and_subtree(external_peer_node.left)
        
        #Checks if both nodes have a right node, to continue the recursion
        if self.right and external_peer_node.right:
            self.right.__do_union_between_copy_self_and_subtree(external_peer_node.right)
    

    def __set_and_get_graph_operated_from_subtrees(self) -> VicinityGraph:
        graph = None
        left_graph = None
        right_graph = None
        
        if self.left.is_leaf():
            if self.left.graph:
                left_graph = self.left.graph
            else:
                raise Exception('Null vicinity graphs in leaves')
        else:
            left_graph = self.left.__set_and_get_graph_operated_from_subtrees()
        
        if self.right.is_leaf():
            if self.right.graph:
                right_graph = self.right.graph
            else:
                raise Exception('Null vicinity graphs in leaves')
        else:
            right_graph = self.right.__set_and_get_graph_operated_from_subtrees()

        if left_graph and right_graph:
            if left_graph.subquery == right_graph.subquery:
                graph = left_graph
            elif self.value == 'AND':
                graph = left_graph.get_intersection_to_graph(right_graph)
            elif self.value == 'OR':
                graph = left_graph.get_union_to_graph(right_graph)
            
            #The new graph inherits the same subquery from the self graph
            if self.graph:
                graph.subquery = self.graph.subquery

            self.graph = graph
            return graph
    

    def __inorder_traversal(self, level: int = 0) -> list[str]:
        result = []

        # Traverse left subtree
        if self.left:
            # Add opening parenthesis if not at root level
            if level > 0: result.append('(')
            # Recursively traverse left subtree and increase level
            result.extend(self.left.__inorder_traversal(level + 1))

        # Append current node value with spaces if it's 'AND' or 'OR'
        if self.value in ['AND', 'OR']:    
            result.append(f' {self.value} ')
        else:
            result.append(self.value)

        # Traverse right subtree
        if self.right:
            # Recursively traverse right subtree and increase level
            result.extend(self.right.__inorder_traversal(level + 1))
            # Add closing parenthesis if not at root level
            if level > 0: result.append(')')
        return result
    

    def __generate_tree_str(self, level: int = 0) -> str:
        output = "\t" * level + self.value + "\n"
        if self.left:
            output += self.left.__generate_tree_str(level + 1)
        if self.right:
            output += self.right.__generate_tree_str(level + 1)
        return output



class BinaryExpressionTree:
    def __init__(self, raw_query: str):
        self.__raw_query = raw_query
        self.root: BinaryTreeNode | None = None
        
        try:
            tokens = self.__separate_boolean_query(self.__raw_query)
            infix_tokens = self.__process_infix_tokens(tokens)
            postfix_tokens = self.__infix_to_postfix(infix_tokens)
            self.__construct_tree_from_postfix(postfix_tokens)
        except Exception as e:
            print('Error initializing BinaryExpressionTree instance: ' + repr(e))
    

    def get_raw_query(self) -> str:
        return self.__raw_query
    

    def get_query_terms_str_list_with_underscores(self) -> list[str]:
        if not self.__check_root_initialized():
            return []
        return self.root.get_values_from_leaves()
    

    def get_query_terms_as_leaves(self) -> list[BinaryTreeNode]:
        if not self.__check_root_initialized():
            return []
        return self.root.get_leaves()
    

    def get_graph(self) -> VicinityGraph | None:
        if not self.__check_root_initialized():
            return
        return self.root.graph
    

    def get_graph_by_subquery(self, query: str) -> VicinityGraph | None:
        if not self.__check_root_initialized():
            return
        
        graph = self.root.get_graph_from_subtree_by_subquery(query)
        if not graph:
            print('Could not find graph for subquery')
            return None
        return graph
    

    def get_union_of_trees(self, 
            query_trees_list: list['BinaryExpressionTree']
            ) -> 'BinaryExpressionTree':
        """
        Get the union between the given list of query trees.

        Parameters
        ----------
        query_trees_list : list[BinaryExpressionTree]
            The list of query trees to be united between

        Returns
        -------
        union_of_trees : BinaryExpressionTree
            The union between the query trees
        """
        # reduce() applies a function of two arguments cumulatively to the items of a sequence or 
        #iterable, from left to right, so as to reduce the iterable to a single value. 
        #For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5)
        union_of_trees = reduce((lambda tree1, tree2: tree1.get_union_to_tree(tree2)), query_trees_list)
        return union_of_trees
    

    def get_union_to_tree(self,
            external_tree: 'BinaryExpressionTree'
            ) -> 'BinaryExpressionTree':
        """
        Unites an external tree with the own tree and obtains a new tree.
        The merging process involves iterating through the graphs of both tree, calculating 
        the sum of weights and the average distances between each one, that is, the union
        between each graph (tree node) of both trees.
        
        Parameters
        ----------
        external_tree : BinaryExpressionTree
            The external tree to be united

        Returns
        -------
        copy_tree : BinaryExpressionTree
            The union between copy of the tree itself and an external tree
        """
        copy_tree = copy.deepcopy(self)
        
        #Union between self tree and None is self tree
        if not external_tree.root:
            return copy_tree
        
        #Union between external tree and None is external tree
        if external_tree.root and not self.root:
            return copy.deepcopy(external_tree)
        
        #Get the deep union between the copy root node and the external root node
        copy_root = copy_tree.root.get_union_to_subtree(external_tree.root)
        copy_tree.root = copy_root
        return copy_tree
    

    def operate_graphs_from_leaves(self) -> None:
        if not self.__check_root_initialized():
            return
        self.root.do_graph_operation_from_subtrees()
    

    def do_text_transformations_to_query_terms(self, 
            stop_words_list: list[str] = [], 
            lema: bool = True, 
            stem: bool = True
            ) -> None:
        """
        Apply text transformations to the terms in the leaf nodes.

        Parameters
        ----------
        stop_words_list : list[str], optional
            List of stop words to be removed from the terms
        lema : bool, optional
            If True, lemmatization is applied
        stem : bool, optional
            If True, stemming is applied
        """
        def transform_node_if_leaf(node: BinaryTreeNode) -> None:
            if node.is_leaf():
                node.value = TextUtils.get_transformed_text_if_it_has_underscores(node.value, stop_words_list, 
                                                                                  lema, stem)
            else:
                transform_node_if_leaf(node.left)
                transform_node_if_leaf(node.right)

        if not self.__check_root_initialized():
            return
        transform_node_if_leaf(self.root)
    

    def initialize_graph_for_each_node(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_query_terms: bool = True,
            summarize: str = 'mean'
            ) -> None:
        """
        Initialize the graph associated to each node in the tree.

        Parameters
        ----------
        nr_of_graph_terms : int, optional
            Configured number of terms in the graph
        limit_distance : int, optional
            Maximal distance of terms used to calculate the vicinity
        include_query_terms : bool, optional
            If True, the query term is included in the vicinity
        summarize : str, optional
            Summarization type to operate distances in the vicinity matrix for each 
            sentence (it can be: mean or median)
        """
        def initialize_graph(node: BinaryTreeNode):
            node.graph = VicinityGraph(str(node), nr_of_graph_terms, limit_distance, 
                                       include_query_terms, summarize)
            if not node.is_leaf():
                initialize_graph(node.left)
                initialize_graph(node.right)
        
        if not self.__check_root_initialized():
            return
        initialize_graph(self.root)
    

    def remove_graphs_for_each_node(self) -> None:
        """
        Remove the graph associated to each node in the tree.
        """
        def remove_graph(node: BinaryTreeNode):
            node.graph = None
            if node.left:
                remove_graph(node.left)
            if node.right:
                remove_graph(node.right)
        
        if not self.__check_root_initialized():
            return
        remove_graph(self.root)


    def __str__(self) -> str:
        if not self.__check_root_initialized():
            return ''
        return str(self.root)
    

    def tree_str(self) -> str:
        if not self.__check_root_initialized():
            return ''
        return self.root.tree_str()
    

    def __separate_boolean_query(self, query: str) -> list[str]:
        # Define a regex pattern to match boolean operators, parentheses, colons and terms
        pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\(|\)|\w+|:)'
        
        # Find all matches using the pattern
        tokens = re.findall(pattern, query)
        
        return tokens
    

    def __process_infix_tokens(self, tokens: list[str]) -> list[str]:
        validation_tuple = self.__validate_boolean_expression(tokens)
        if validation_tuple[0]:
            tokens_deleted_not_expressions = self.__process_boolean_tokens_delete_not_expressions(tokens)
            tokens_deleted_colon_and_keys = self.__process_boolean_tokens_delete_colon_and_keys(tokens_deleted_not_expressions)
            infix_tokens = self.__join_term_tokens_by_underscores(tokens_deleted_colon_and_keys)
            return infix_tokens
        else:
            raise Exception(validation_tuple[1])
    

    def __infix_to_postfix(self, infix_tokens: list[str]) -> list[str]:
        # Operator precedence
        precedence = {'OR': 1, 'AND': 2}
        # Stack for operators
        operator_stack = deque()
        # List for expression in postfix
        postfix_list = []

        for token in infix_tokens:
            if token in precedence.keys():
                # If the token is an operator, handle precedence (operator_stack[0] is the top element of stack)
                while (operator_stack and operator_stack[0] != '(' and
                    precedence[operator_stack[0]] >= precedence[token]):
                    postfix_list.append(operator_stack.popleft())
                operator_stack.appendleft(token)
            elif token == '(':
                # If the token is '(', add it to the operator stack
                operator_stack.appendleft(token)
            elif token == ')':
                # If the token is ')', pop until find '(' (operator_stack[0] is the top element of stack)
                while operator_stack and operator_stack[0] != '(':
                    postfix_list.append(operator_stack.popleft())
                operator_stack.popleft()  # Pop '(' from stack
            else:
                # If the token is an operand (alphanumeric word or containing '_'), add it to postfix
                postfix_list.append(token)

        # Add any remaining operators on the stack to the postfix list
        while operator_stack:
            postfix_list.append(operator_stack.popleft())

        return postfix_list
    
    
    def __construct_tree_from_postfix(self, postfix_tokens: list[str]) -> None:
        stack = []
        for token in postfix_tokens:
            if token in ['AND', 'OR']:
                # If the token is an operator, create a node and link the subtrees
                node = BinaryTreeNode(token)
                node.right = stack.pop()
                node.left = stack.pop()
                stack.append(node)
            else:
                # If the token is an operand, create a node and add it to the stack
                node = BinaryTreeNode(token)
                stack.append(node)
        
        # The last element on the stack is the root of the tree
        self.root = stack.pop()


    def __validate_boolean_expression(self, tokens: list[str]) -> tuple[bool, str]:
        # Check parenthesis syntax
        stack = []
        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                if not stack:
                    return False, "Unmatched closing parenthesis"
                stack.pop()
        if stack:
            return False, "Unmatched opening parenthesis"

        # Check AND OR operators
        operators = {'AND', 'OR'}
        length = len(tokens)
        
        for i, token in enumerate(tokens):
            if token in operators:
                if i == 0 or i == length - 1:
                    return False, f"Operator {token} cannot be at the start or end of the expression"
                if tokens[i-1] in operators or tokens[i+1] in operators:
                    return False, f"Operator {token} must be surrounded by valid terms"

        return True, "The expression is valid"


    def __process_boolean_tokens_delete_not_expressions(self, tokens: list[str]) -> list[str]:
        indices_to_exclude = []
        index = 0

        while index < len(tokens):
            next_index = index + 1
            if tokens[index] == 'NOT':
                stack = 0
                while next_index < len(tokens):
                    if tokens[next_index] == '(':
                        stack += 1
                    elif tokens[next_index] == ')':
                        if stack < 2: # if it is the last closing parenthesis, or there's no opening parenthesis between the 2 indices
                            if stack == 1:
                                next_index += 1     # removes the closing parenthesis if there was an opening parenthesis before
                            break
                        else:
                            stack -= 1
                    elif tokens[next_index] in ['AND', 'OR', 'NOT']:
                        if stack < 1:   # if there was no opening parenthesis between the two indices
                            break
                    next_index += 1
                indices_to_exclude.extend(range(index, next_index))  # add the indexes to delete their values
            index = next_index
        
        processed_tokens = [token for index, token in enumerate(tokens) if index not in indices_to_exclude]
        return processed_tokens


    def __process_boolean_tokens_delete_colon_and_keys(self, tokens: list[str]) -> list[str]:
        prev_index = 0
        indices_to_remove = []
        for index, token in enumerate(tokens):
            if token == ':':
                prev_index = index
                while prev_index > 0:
                    if tokens[prev_index - 1] in ['AND', 'OR', '(']:
                        break
                    else:
                        prev_index -= 1
                indices_to_remove.extend(range(prev_index, index+1))  # add the indexes to delete their values
        processed_tokens = [token for index, token in enumerate(tokens) if index not in indices_to_remove]
        return processed_tokens


    def __join_term_tokens_by_underscores(self, tokens: list[str]) -> list[str]:
        if len(tokens) <= 1:
            return tokens
        else:
            _2d_matrix_to_join_by_underscore = []
            for index, token in enumerate(tokens):
                if (token in ['AND', 'OR', ')']) or (index == (len(tokens)-1)): #if token is operator or its the last element
                    prev_index = index - 1
                    while prev_index >= 0:
                        if tokens[prev_index] in ['AND', 'OR', '(', ')']:
                            prev_index += 1
                            break
                        if prev_index == 0:
                            break
                        prev_index -= 1
                    last_index = index - 1 if token in ['AND', 'OR', ')'] else index
                    tuple_to_join_by_underscore = (prev_index, last_index)
                    if (tuple_to_join_by_underscore[1] - tuple_to_join_by_underscore[0]) > 0:
                        _2d_matrix_to_join_by_underscore.append(tuple_to_join_by_underscore)
            
            current_index = 0
            processed_tokens = []
            for start, end in _2d_matrix_to_join_by_underscore:
                # Make sure indexes are valid
                if start < 0 or end >= len(tokens) or start > end:
                    raise ValueError(f"Invalid indexes for joining elements: {start}, {end}")

                # Add the token elements before the current subset
                while current_index < start:
                    processed_tokens.append(tokens[current_index])
                    current_index += 1

                # Join subset elements with underscores
                joined_element = '_'.join(tokens[start:end+1])
                processed_tokens.append(joined_element)

                # Move the current index to the end of the subset
                current_index = end + 1
            
            # Agregar los elementos restantes de tokens después del último subconjunto
            while current_index < len(tokens):
                processed_tokens.append(tokens[current_index])
                current_index += 1       

        return processed_tokens
    

    def __check_root_initialized(self) -> bool:
        if not self.root:
            print('Error initializing BinaryExpressionTree instance')
            return False
        return True



class QueryTreeHandler:
    def __init__(self, query_tree):
        self.__query_tree: BinaryExpressionTree = query_tree


    def get_query_tree(self) -> BinaryExpressionTree:
        return self.__query_tree
    

    def set_query_tree(self, query_tree) -> None:
        self.__query_tree = query_tree


    def get_graph(self) -> VicinityGraph | None:
        return self.__query_tree.get_graph()


    def get_graph_by_subquery(self, query: str) -> VicinityGraph | None:
        return self.__query_tree.get_graph_by_subquery(query)



class Sentence(QueryTreeHandler):
    
    def __init__(self, raw_text: str, query: BinaryExpressionTree, position_in_doc: int = 0, weight: float = 1.0):
        super().__init__(query_tree=query)
        self.__raw_text = raw_text
        self.__position_in_doc = position_in_doc
        self.__weight = weight
        self.__preprocessed_text: str = ""
        self.__vicinity_matrix: dict[str, dict[str, list[float]]] = []


    def get_raw_text(self) -> str:
        return self.__raw_text


    def get_position_in_doc(self) -> int:
        return self.__position_in_doc
    

    def get_weight(self) -> float:
        return self.__weight


    def get_preprocessed_text(self) -> str:
        return self.__preprocessed_text
    

    def set_preprocessed_text(self, value: str) -> None:
        self.__preprocessed_text = value
    

    def get_vicinity_matrix(self) -> dict[str, dict[str, list[float]]]:
        return self.__vicinity_matrix


    def do_text_transformations_if_any_query_term(self,
            stop_words_list: list[str], 
            lema: bool = True, 
            stem: bool = True
            ) -> None:
        """
        Apply some text transformations to the sentence and calculate term positions to the 
        sentence. Also replaces underscores with spaces from the query terms.

        Parameters
        ----------
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool, optional
            If True, lematization is applied
        stem : bool, optional
            If True, stemming is applied
        """
        # Sentence query graphs is re-initialized
        self.get_query_tree().remove_graphs_for_each_node()
        
        transformed_sentence_str = TextUtils.get_transformed_text(self.__raw_text, stop_words_list, lema, stem)
        query_terms_with_underscores = self.get_query_tree().get_query_terms_str_list_with_underscores()
        query_terms_with_spaces = [term.replace('_', ' ') for term in query_terms_with_underscores]

        #If there is any query term in the transformed sentence string
        if any(query_term in transformed_sentence_str for query_term in query_terms_with_spaces):  
            transformed_sentence_str = self.__get_transformed_sentence_str_with_underscores_in_query_terms(query_terms_with_underscores, 
                                                                                                    transformed_sentence_str)
            self.__preprocessed_text = transformed_sentence_str
    

    def calculate_vicinity_matrix(self) -> None:
        """
        Calculate term positions dictionary for terms of the sentence and terms from the query terms, 
        then it calculates the vicinity of a list of query terms in the current sentence, limited by 
        a specified distance.

        E.g. {'v_term_1': {'query_term_1': [0.0, 1.8, 0.0, 0.9]}, 
        'v_term_2': {'query_term_1': [0.0, 0.0, 0.9, 0.0], 'query_term_2': [2.7, 0.0, 0.0, 0.9]}, ... }
        """
        term_positions_dict = self.get_term_positions_dict(self.__preprocessed_text)
        query_term_positions_dict = self.get_query_term_positions_dict(term_positions_dict, 
                                                            self.get_query_tree().get_query_terms_str_list_with_underscores())
        vicinity_matrix = {}  # Create the empty dictionary

        # Calculate all terms in term_positions_defaultdict that are at distance limit_distance (or closer) to the query_terms
        # and return a list of these terms and their corresponding distances
        for term, term_positions in term_positions_dict.items():
            #Avoid comparing the query term with itself (if bool false)
            if((term not in query_term_positions_dict.keys()) or (self.get_graph().get_config().get_include_query_terms())): 
                # Calculate the distance between the query term and the rest of terms
                first_one = True
                # Iterate query terms that do not contain spaces
                for query_term, query_positions in query_term_positions_dict.items():
                    if query_term != term:
                        freq_neighborhood_positions = self.calculate_ponderation_of_distances_between_term_positions(query_positions, 
                                                                                                                       term_positions,
                                                                                    self.get_graph().get_config().get_limit_distance(),
                                                                                                                        self.__weight)

                        if (any(frq > 0 for frq in freq_neighborhood_positions)):
                            if (first_one):
                                vicinity_matrix[term] = {}
                                first_one = False
                            vicinity_matrix[term][query_term] = freq_neighborhood_positions

        self.__vicinity_matrix = vicinity_matrix
    

    def generate_nodes_in_sentence_graphs(self) -> None:
        """
        Generate nodes to all the graphs associated with the sentence, based on 
        their isolated query terms as leaves from the query tree.
        """
        #First, generate nodes to the graphs associated with the leaves from the query tree
        self.generate_nodes_in_all_leaf_graphs()
        
        #Then, generate nodes to the graphs associated with the rest of the nodes in the tree
        self.get_query_tree().operate_graphs_from_leaves()
    

    def generate_nodes_in_all_leaf_graphs(self):
        """
        Generate nodes to the graphs associated with the leaves from the query tree
        """
        for leaf_node in self.get_query_tree().get_query_terms_as_leaves():
            self.generate_nodes_in_leaf_graph(leaf_node)
    

    def get_term_positions_dict(self, 
            text: str
            ) -> defaultdict[str, list[int]]:
        """
        Calculate a dictionary to store the list of positions for each text term.

        Parameters
        ----------
        text : str
            The text to calculate its positions by term

        Returns
        -------
        term_positions_dict : defaultdict[str, list[int]]
            A dictionary with a list of positions for each term of the text
        """
        vectorizer = CountVectorizer()
        vector = vectorizer.build_tokenizer()(text)
        term_positions_dict = defaultdict(list)
        for i in range(len(vector)):
            term_positions_dict[vector[i]].append(i)
        return term_positions_dict
    

    def get_query_term_positions_dict(self, 
            term_positions_dict: defaultdict[str, list[int]],
            query_terms: list[str]
            ) -> dict[str, list[int]]:
        """
        Calculate a dictionary to store the list of positions for each query term, based on 
        the positions of each term given the term positions dict.

        Parameters
        ----------
        term_positions_dict : defaultdict[str, list[int]]
            A dictionary with a list of positions from a text
        query_terms : list[str]
            List of query terms to get their positions

        Returns
        -------
        query_term_positions_dict : dict[str, list[int]]
            A dictionary with a list of positions for each term of the query inside the 
            term positions dict
        """
        query_term_positions_dict = {}

        for query_term in query_terms:
            # If the query term is within the term position dictionary
            if query_term in term_positions_dict.keys():
                # Get the term positions of the query term
                query_term_positions_dict[query_term] = term_positions_dict[query_term]
        
        return query_term_positions_dict


    def calculate_ponderation_of_distances_between_term_positions(self,
            term1_positions: list[int], 
            term2_positions: list[int],
            limit_distance: int,
            weight: float = 1.0
            ) -> list[float]:
        """
        Compare the positions vectors of two terms, and return the ponderation 
        (frequency * weight) list per distance between query terms and vicinity terms
        
        E.g.
        term1_positions = [0, 2, 4, 6]
        term2_positions = [1, 3, 5, 7]
        limit_distance = 7
        weight = 1.0
        result = [7.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0]

        Parameters
        ----------
        term1_positions : list[int]
            List of positions of the first term
        term2_positions : list[int]
            List of positions of the second term
        limit_distance : int
            Limit distance to calculate between the two term positions lists
        weight : float, optional
            Weight to multiply with each frequency, to get its ponderation

        Returns
        -------
        ponderations_per_distance : list[float]
            List of ponderations per distance between query terms and vicinity terms
        """
        ponderations_per_distance = [0] * limit_distance

        for term1_pos in term1_positions:
            for term2_pos in term2_positions:
                absolute_distance = abs(term1_pos-term2_pos)
                if (absolute_distance <= limit_distance):
                    ponderations_per_distance[absolute_distance-1] += 1
        
        ponderations_per_distance = [i * weight for i in ponderations_per_distance]
        return ponderations_per_distance
    

    def __get_transformed_sentence_str_with_underscores_in_query_terms(self,
            query_terms_with_underscores: list[str],
            transformed_sentence_str: str) -> str:
        """
        Get the transformed sentence string with underscores in the query terms.

        Parameters
        ----------
        query_terms_with_underscores : list[str]
            List of operands from the query
        transformed_sentence_str : str
            Transformed sentence string to replace spaces with underscores

        Returns
        -------
        sentence_str_with_underscores_in_query_terms : str
            Transformed sentence string without spaces in query terms within the sentence
        """
        # If the query terms do not have underscores, then the same transformed sentence is returned
        sentence_str_with_underscores_in_query_terms = transformed_sentence_str
        
        for query_term_with_underscores in query_terms_with_underscores:
            query_term_with_spaces = query_term_with_underscores.replace('_', ' ')
            #If the query term is in the transformed sentence string and the query term contains more than a word
            if (query_term_with_spaces in transformed_sentence_str) and ("_" in query_term_with_underscores): 
                sentence_str_with_underscores_in_query_terms = transformed_sentence_str.replace(query_term_with_spaces, 
                                                                                                query_term_with_underscores)
        return sentence_str_with_underscores_in_query_terms
    

    def generate_nodes_in_leaf_graph(self, 
            leaf_node: BinaryTreeNode,
            ) -> None:
        """
        Generate nodes for the graph associated with the sentence, based on the isolated query term of 
        the leaf from the query tree. This method calculates the ponderations of the terms associated 
        with the query term of the graph, and adds each of them to their respective node.

        Parameters
        ----------
        leaf_node : BinaryTreeNode
            A leaf node associated with the sentence tree
        """
        query_term = leaf_node.value  
        terms_pond_dict = self.get_terms_ponderation_dict(query_term)

        # Iterate over each term in the ponderation dictionary
        for neighbor_term in terms_pond_dict.keys():
            # Retrieve the list of ponderations by distance for the term
            list_of_pond_by_distance = self.__vicinity_matrix.get(neighbor_term).get(query_term)
            distance_calculation_list = []

            # Construct a list of distances multiplied by its frequencies
            for idx, freq_mult_by_weight in enumerate(list_of_pond_by_distance):
                frequency = round(freq_mult_by_weight / self.__weight)
                distance_calculation_list.extend([idx+1] * frequency)
            
            #Summarize the list of distances based on the graph settings (it can be: mean or median)
            if self.get_graph().get_config().get_summarize() == 'median':
                _distance = np.median(distance_calculation_list)
            else:
                _distance = np.mean(distance_calculation_list)
                
            # Calculate the mean distance for the term
            new_node = VicinityNode(term=neighbor_term, ponderation=terms_pond_dict.get(neighbor_term), distance=_distance)
            leaf_node.graph.add_node(new_node)

    

    def get_terms_ponderation_dict(self, 
            query_term: str
            ) -> dict[str, float]:
        """
        This method calculates the ponderation (frequency * weight) of terms from the 
        vicinity matrix of the sentence, by a specified query term

        Parameters
        ----------
        query_term : str
            The query term of the graph

        Returns
        -------
        terms_pond_dict : dict[str, float]
            A dictionary containing the ponderation of terms. Each dictionary has keys corresponding 
            to terms, and values represent the ponderation of the term within the sentence.

        """
        terms_pond_dict = {}

        # Iterate over each term and its ponderations in the document
        for neighbor_term, distance_pond_by_query_term in self.__vicinity_matrix.items():
            # Checks if the query word is in the distance-ponderation 
            # dictionary keys and if the neighbor term isn't a query term
            if (query_term in distance_pond_by_query_term):
                sum_of_ponds_in_query_term = sum(distance_pond_by_query_term[query_term])
                if sum_of_ponds_in_query_term > 0:
                    terms_pond_dict[neighbor_term] = sum_of_ponds_in_query_term
            
        return terms_pond_dict



class Document(QueryTreeHandler):
    
    def __init__(self, query: BinaryExpressionTree, abstract: str = "", title: str = "", 
                 doc_id: str = "1", weight: float = 1.0, ranking_position: int = 1):
        super().__init__(query_tree=query)
        self.__abstract = abstract
        self.__title = title
        self.__doc_id = doc_id
        self.__weight = weight
        self.__ranking_position = ranking_position
        self.__sentences: list[Sentence] = []

        self.__calculate_sentences_list_from_documents()
    

    def get_abstract(self) -> str:
        return self.__abstract
    

    def get_title(self) -> str:
        return self.__title
    

    def get_doc_id(self) -> str:
        return self.__doc_id
    

    def get_sentences(self) -> list[Sentence]:
        return self.__sentences
    

    def get_weight(self) -> float:
        return self.__weight
    

    def get_ranking_position(self) -> int:
        return self.__ranking_position
    

    def get_sentence_by_raw_text(self, text: str) -> Sentence | None:
        for sentence in self.__sentences:
            if sentence.get_raw_text() == text:
                return sentence
        print(f'No sentence with raw text: {text}')
        return None
    

    def get_sentence_by_position_in_doc(self, position: int) -> Sentence | None:
        for sentence in self.__sentences:
            if sentence.get_position_in_doc() == position:
                return sentence
        print(f'No sentence with position: {position}')
        return None
    

    def get_list_of_query_trees_from_sentences(self) -> list[BinaryExpressionTree]:
        """
        Returns a list of query trees from the sentences of the current document.
        """
        list_of_query_trees = [sentence.get_query_tree() for sentence in self.__sentences]
        return list_of_query_trees
    

    def do_text_transformations_by_sentence(self,
            stop_words_list: list[str], 
            lema: bool,
            stem: bool
            ) -> None:
        """
        Apply some text transformations to each sentence of the document 
        and calculate term positions to the sentence.

        Parameters
        ----------
        stop_words_list : list[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        """
        # Document query graphs is re-initialized
        self.get_query_tree().remove_graphs_for_each_node()
        
        for sentence in self.__sentences:
            sentence.do_text_transformations_if_any_query_term(stop_words_list, lema, stem)


    def generate_graph_nodes_of_doc_and_sentences(self) -> None:
        """
        Generate all the nodes associated with the document graphs, along with 
        the graphs of their sentences, based on their query trees.
        """
        #Generate graph nodes in sentences
        for sentence in self.__sentences:
            sentence.generate_nodes_in_sentence_graphs()
        
        #Generate graph nodes in the current document
        self.set_query_tree(self.__get_union_of_sentences_trees())
    

    def calculate_vicinity_matrix_of_sentences(self) -> None:
        """
        Calculate term positions dictionary for terms of the sentence and terms from
        the query terms, then it calculates the vicinity of a list of query 
        terms in the current document, limited by a specified distance.
        """
        for sentence in self.__sentences:
            sentence.calculate_vicinity_matrix()


    def __calculate_sentences_list_from_documents(self) -> None:
        """
        Transform the text of each document in the input list into a list of sentences. Split 
        the text by dots. It also generates a copy of the document tree and adds it as an 
        attribute of the new sentence.
        """
        self.__sentences = []
        list_of_sentence_str = self.__get_list_of_sentence_strings()
        for index, sentence_str in enumerate(list_of_sentence_str):
            query_copy = copy.deepcopy(self.get_query_tree())
            sentence_obj = Sentence(raw_text=sentence_str, query=query_copy, 
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
        list_of_sentence_str = []
        if self.__title:
            list_of_sentence_str.append(self.__title)
        if self.__abstract:
            abstract_list_of_sentence_str = self.__abstract.split('. ')
            list_of_sentence_str.extend(abstract_list_of_sentence_str)
        return list_of_sentence_str
    

    def __get_union_of_sentences_trees(self) -> BinaryExpressionTree:
        """
        Get the union between the list of query trees associated with the sentences of the document.

        Returns
        -------
        union_of_trees : BinaryExpressionTree
            The union between the sentence query trees.
        """
        query_trees_list = self.get_list_of_query_trees_from_sentences()
        union_of_trees = self.get_query_tree().get_union_of_trees(query_trees_list)
        return union_of_trees



class TextTransformationsConfig:
    def __init__(self, stop_words: list[str] = [], lemmatization: bool = True, stemming: bool = False):
        self.__stop_words = stop_words
        self.__lemmatization = lemmatization
        self.__stemming = stemming
    

    def get_transformations_params(self) -> tuple[list[str], bool, bool]:
        return self.__stop_words, self.__lemmatization, self.__stemming



class Ranking(QueryTreeHandler):
    
    def __init__(self, query_text: str, nr_search_results: int = 10, ranking_weight_type: str = 'linear', 
                 stop_words: list[str] = [], lemmatization: bool = True, stemming: str = False):
        super().__init__(query_tree=BinaryExpressionTree(query_text))
        self.__nr_search_results = nr_search_results
        self.__ranking_weight_type = ranking_weight_type  #Type of weighting to be applied (it can be: 'none', 'linear' or 'inverse')
        self.__text_transformations_config = TextTransformationsConfig(stop_words, lemmatization, stemming)
        self.__documents: list[Document] = []

        self.__do_text_transformations_to_query_terms()


    def get_documents(self) -> list[Document]:
        return self.__documents 
    

    def get_text_transformations_config(self) -> TextTransformationsConfig:
        return self.__text_transformations_config
    

    def get_document_by_title(self, title: str) -> Document | None:
        for document in self.__documents:
            if document.get_title() == title:
                return document
        print(f'No document with title: {title}')
        return None
    

    def get_document_by_id(self, id: str) -> Document | None:
        for document in self.__documents:
            if document.get_doc_id() == id:
                return document
        print(f'No document with id: {id}')
        return None
    

    def get_document_by_ranking_position(self, position: int) -> Document | None:
        for document in self.__documents:
            if document.get_ranking_position() == position:
                return document
        print(f'No document with ranking position: {position}')
        return None
    

    def get_list_of_query_trees_from_documents(self) -> list[BinaryExpressionTree]:
        """
        Returns a list of query trees from the documents of the current ranking.
        """
        list_of_query_trees = [document.get_query_tree() for document in self.__documents]
        return list_of_query_trees
    

    def calculate_article_dictionaries_list(self, 
            articles_dicts: list[dict]
            ) -> None:
        """
        Calculate the current ranking from a list of article dictionaries ordered by their 
        ranking position. Each dictionary must contain the attributes 'title', 'abstract'
        and 'article_number'.

        Transforms the list of article dictionaries into a list of Document type objects, 
        which have a weight associated with their position in the received list. 
        Objects will be appended to the ranking. In addition, pre-processing of the text 
        of the documents is carried out.

        Parameters
        ----------
        nr_of_graph_terms : int
            Configured number of terms in the graph
        """
        self.__calculate_ranking_as_weighted_documents_and_do_text_transformations(articles_dicts)
    
    
    def calculate_ieee_xplore_ranking(self) -> None:
        """
        Calculate the IEEE-Xplore ranking for the current ranking. 

        Transforms the list of article dictionaries into a list of Document type objects, 
        which have a weight associated with their position in the received list. 
        Objects will be appended to the ranking. In addition, pre-processing of the text 
        of the documents is carried out.
        """
        articles_dicts = self.__get_ieee_xplore_ranking()
        self.__calculate_ranking_as_weighted_documents_and_do_text_transformations(articles_dicts)
    

    def generate_all_graphs(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_query_terms: bool = True,
            summarize: str = 'mean'
            ) -> None:
        """
        Initialize graphs associated to all the sentences of each document from 
        the ranking. Sentences have no graph term limit. Also calculates the 
        vicinity matrix for each sentence. Finally, generates nodes of all graphs.

        Parameters
        ----------
        nr_of_graph_terms : int, optional
            Configured number of terms in the graph
        limit_distance : int, optional
            Maximal distance of terms used to calculate the vicinity
        include_query_terms : bool, optional
            If True, the query term is included in the vicinity
        summarize : str, optional
            Summarization type to operate distances in the vicinity matrix for 
            each sentence (it can be: mean or median)
        """
        #Set graph attributes to the graphs of each document and the sentences' graphs of each document
        self.initialize_graphs_for_all_trees(nr_of_graph_terms, limit_distance, 
                                               include_query_terms, summarize)

        #Calculate term positions and vicinity matrix of each sentence by document
        self.calculate_vicinity_matrix_of_sentences_by_doc()
        
        #Generate nodes of all graphs
        self.generate_nodes_of_all_graphs()
    

    def get_ieee_xplore_article(self,
            parameter: str, 
            value: str
            ) -> dict:
        """
        Get an article from IEEE-Xplore.
        
        Parameters
        ----------
        parameter: str
            Parameter used to search the article (e.g. 'article_number')
        value: str
            Value of the parameter used to search the article (e.g. '8600704')
        
        Returns
        -------
        article : dict
            The IEEE Xplore article
        """
        xplore_id = '6g7w4kfgteeqvy2jur3ak9mn'
        query = XPLORE(xplore_id)
        query.outputDataFormat='object'
        query.addParameter(parameter, value)
        data = query.callAPI()
        article = {}
        article["abstract"] = data.get('articles', [{}])[0].get('abstract', "")
        article["title"] = data.get('articles', [{}])[0].get('title', "")
        article["article_number"] = data.get('articles', [{}])[0].get('article_number', "1")

        return article
    

    def initialize_graphs_for_all_trees(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_query_terms: bool = True,
            summarize: str = 'mean'
            ) -> None:
        """
        Generate all the nodes associated with the graphs from both the ranking and all 
        the documents along with their sentences, based on the query terms.
        """
        parameters_tuple = (nr_of_graph_terms, limit_distance, include_query_terms, summarize)

        #Initialize the graphs of the query tree associated with the ranking
        self.get_query_tree().initialize_graph_for_each_node(*parameters_tuple)

        #Initialize the graphs of the query trees associated with the documents of the ranking
        for document in self.__documents:
            document.get_query_tree().initialize_graph_for_each_node(*parameters_tuple)
        
        #Initialize the graphs of the query trees associated with the sentences from the documents of the ranking
        for document in self.__documents:
            for sentence in document.get_sentences():
                sentence.get_query_tree().initialize_graph_for_each_node(*parameters_tuple)
    

    def calculate_vicinity_matrix_of_sentences_by_doc(self) -> None:
        """
        Calculate term positions dictionary for terms of each sentence and terms from
        the query terms, then it calculates the vicinity of a list of query 
        terms in all documents, limited by a specified distance.
        """
        for document in self.__documents:
            document.calculate_vicinity_matrix_of_sentences()
    

    def generate_nodes_of_all_graphs(self) -> None:
        """
        Generate all the nodes associated with the graphs from both the ranking and all 
        the documents along with their sentences, based on the query terms.
        """
        #Generate graph nodes of documents and its sentences
        for document in self.__documents:
            document.generate_graph_nodes_of_doc_and_sentences()
        
        #Then, generate nodes of the ranking class query tree
        self.set_query_tree(self.__get_union_of_documents_trees())
    

    def __get_union_of_documents_trees(self) -> BinaryExpressionTree:
        """
        Get the union between the list of query trees associated with the 
        documents of the ranking.

        Returns
        -------
        union_of_trees : BinaryExpressionTree
            The union between the document query trees.
        """
        query_trees_list = self.get_list_of_query_trees_from_documents()
        union_of_trees = self.get_query_tree().get_union_of_trees(query_trees_list)
        return union_of_trees


    def __do_text_transformations_to_query_terms(self) -> None:
        """
        Apply text transformations to the query terms of the ranking. Lower the text, tokenize, remove 
        punctuation, stopwords, finally do stemming and lemmatization if specified.
        """
        self.get_query_tree().do_text_transformations_to_query_terms(*self.__text_transformations_config.get_transformations_params())


    def __get_ieee_xplore_ranking(self) -> list[dict]:
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
        query.queryText(self.get_query_tree().get_raw_query())
        data = query.callAPI()
        results = data.get('articles', [{}])
        
        return results


    def __calculate_ranking_as_weighted_documents_and_do_text_transformations(self,
            articles_dicts: list[dict],
            ) -> None:
        """
        Transforms the list of article dictionaries into a list of Document type objects, which have a weight
        associated with their position in the received list. Objects will be appended to the ranking.
        In addition, pre-processing of the text of the documents is carried out.
        If weighted <> none : the document will be weighted depending on its position in the ranking.  

        Parameters
        ----------
        articles_dicts : list[dict]
            List of article dictionaries to be processed as Document type objects and to be part of the ranking
        """
        # Ranking documents and ranking query graphs are re-initialized
        self.__documents = []
        self.get_query_tree().remove_graphs_for_each_node()

        for index, article in enumerate(articles_dicts):
            new_doc = self.__get_new_document_by_article(article, index, results_size=len(articles_dicts))
            new_doc.do_text_transformations_by_sentence(*self.__text_transformations_config.get_transformations_params())
            self.__documents.append(new_doc)
    

    def __get_new_document_by_article(self, 
            article : dict, 
            index : int,
            results_size : int,
            ) -> Document:
        """
        Create and get a new Document object from an article. It also generates a copy 
        of the ranking query tree and adds it as an attribute of the new document.

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
        _abstract = article.get('abstract', "")
        _title = article.get('title', "")
        _doc_id = article.get('article_number', "1")
        _ranking_pos = index + 1
        _weight = self.__calculate_weight(self.__ranking_weight_type, results_size, _ranking_pos)
        _query_copy = copy.deepcopy(self.get_query_tree())
        new_doc = Document(query=_query_copy, abstract=_abstract, title=_title, doc_id=_doc_id, 
                            weight=_weight, ranking_position=_ranking_pos)
        
        return new_doc


    def __calculate_weight(self,
            weighted: str, 
            results_size: int, 
            ranking_position: int
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
        ranking_position : int
            The position of the document in the ranking.

        Returns
        -------
        factor : float
            The calculated weight factor.
        """
        if (weighted=='linear'):
            factor = float((results_size - ((ranking_position - 1) * 0.25)) / results_size)
        elif (weighted=='inverse'):
            factor = float(1 / (((ranking_position - 1) * 0.03) + 1))
        else:
            factor = 1.0

        return factor