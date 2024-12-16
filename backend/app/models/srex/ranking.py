import numpy as np
from collections import defaultdict, Counter
import copy
import re

from models.ieee_xplore.xploreapi import XPLORE
from models.srex.vicinity_graph import VicinityGraph
from models.srex.vicinity_graph import VicinityNode
from models.srex.binary_expression_tree import BinaryExpressionTree
from models.srex.binary_expression_tree import BinaryTreeNode
from utils.text_utils import TextUtils
from utils.math_utils import MathUtils
from utils.vector_utils import VectorUtils




class QueryTreeHandler:
    def __init__(self, query_tree):
        """
        Initialize a new QueryTreeHandler object.

        Parameters:
        query_tree (BinaryExpressionTree): The query tree for the handler.
        """
        self.__query_tree: BinaryExpressionTree = query_tree


    @property
    def query_tree(self) -> BinaryExpressionTree:
        """
        Retrieve the query tree associated with the handler.

        Returns:
        BinaryExpressionTree: The query tree associated with the handler.
        """
        return self.__query_tree


    @query_tree.setter
    def query_tree(self, query_tree: BinaryExpressionTree) -> None:
        """
        Set a new query tree for the handler.

        Parameters:
        query_tree (BinaryExpressionTree): The new query tree to be associated with the handler.
        """
        self.__query_tree = query_tree


    def get_graph(self) -> VicinityGraph | None:
        """
        Retrieve the graph associated with the query tree of the handler.

        Returns:
        VicinityGraph | None: The graph associated with the query tree, or None if the graph does not exist.
        """
        return self.__query_tree.get_graph()


    def get_graph_by_subquery(self, query: str) -> VicinityGraph | None:
        """
        Retrieve the graph associated with a specific subquery within the query tree of the handler.

        Parameters:
        query (str): The subquery for which to retrieve the graph.

        Returns:
        VicinityGraph | None: The graph associated with the subquery, or None if the graph does not exist.
        """
        return self.__query_tree.get_graph_by_subquery(query)



class Sentence(QueryTreeHandler):
    
    def __init__(self, raw_text: str, query: BinaryExpressionTree, position_in_doc: int = 0, weight: float = 1.0):
        """
        Initialize a new Sentence object.

        Parameters:
        raw_text (str): The raw text of the sentence.
        query (BinaryExpressionTree): The query tree associated with the sentence.
        position_in_doc (int, optional): The position of the sentence in the document. Default is 0.
        weight (float, optional): The weight of the sentence in the ranking. Default is 1.0.
        """
        super().__init__(query_tree=query)
        self.__raw_text = raw_text
        self.__position_in_doc = position_in_doc
        self.__weight = weight
        self.__preprocessed_text: str = ""
        self.__vicinity_matrix: dict[str, dict[str, list[int]]] = {}


    @property
    def raw_text(self) -> str:
        """
        Retrieve the raw text of the current sentence.

        Returns:
        str: The raw text of the sentence.
        """
        return self.__raw_text


    @property
    def position_in_doc(self) -> int:
        """
        Retrieve the position of the sentence in the document.

        Returns:
        int: The position of the sentence in the document. The position index is 0-based.
        """
        return self.__position_in_doc


    @property
    def preprocessed_text(self) -> str:
        """
        Retrieve the preprocessed text after applying text transformations.

        Returns:
        str: The preprocessed text after applying text transformations.
        """
        return self.__preprocessed_text
    

    @property
    def vicinity_matrix(self) -> dict[str, dict[str, list[float]]]:
        """
        Retrieve the vicinity matrix of the current Document object. The vicinity matrix is a dictionary 
        where the keys are the terms and the values are dictionaries. Each inner dictionary contains 
        the frequencies of the term by distance from the query term.

        Returns:
        dict[str, dict[str, list[float]]]: The vicinity matrix of the Document object. The outer dictionary 
        has terms as keys, and the inner dictionaries have distances as keys and lists of frequencies 
        as values.
        """
        return self.__vicinity_matrix


    def do_text_transformations(self,
            stop_words: tuple[str] = (), 
            lema: bool = True, 
            stem: bool = False
            ) -> None:
        """
        Apply some text transformations to the sentence. Also replaces spaces in query terms with underscores.

        Parameters
        ----------
        stop_words : tuple[str]
            List of stop words to be removed from the sentence
        lema : bool, optional
            If True, lematization is applied
        stem : bool, optional
            If True, stemming is applied
        """
        # Sentence query graphs is re-initialized
        self.query_tree.remove_graphs_for_each_node()
        
        transformed_sentence_str = TextUtils.get_transformed_text(self.__raw_text, stop_words, lema, stem)
        query_terms_with_underscores = self.query_tree.get_query_terms_str_list_with_underscores() 
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
        term_positions_dict = self.get_term_positions_dict()
        query_terms = self.query_tree.get_query_terms_str_list_with_underscores()
        query_term_positions_dict = self.get_query_term_positions_dict(term_positions_dict, query_terms)
        vicinity_matrix = {}   # Create the empty dictionary
        
        limit_distance = self.get_graph().config.limit_distance
        include_query_terms = self.get_graph().config.include_query_terms

        # Calculate all terms in term_positions_defaultdict that are at distance limit_distance (or closer) to the query_terms
        # and return a list of these terms and their corresponding distances
        for term, term_positions in term_positions_dict.items():
            #Avoid comparing the query term with itself (if bool false)
            if term in query_term_positions_dict and not include_query_terms:
                continue

            term_vicinity = {}
            # Calculate the distance between the query term and the rest of terms
            for query_term, query_positions in query_term_positions_dict.items():
                if query_term == term:
                    continue

                freq_neighborhood_positions = VectorUtils.calculate_distances_between_term_positions(
                    query_positions, term_positions, limit_distance)

                if any(freq > 0 for freq in freq_neighborhood_positions):
                    term_vicinity[query_term] = freq_neighborhood_positions

            if term_vicinity:
                vicinity_matrix[term] = term_vicinity

        self.__vicinity_matrix = vicinity_matrix
    

    def generate_nodes_in_tree_graphs(self) -> None:
        """
        Generate nodes to all the graphs associated with the sentence tree, based 
        on their isolated query terms as leaves from the query tree. Also, generate
        frequency criteria VicinityNodes for the root BinaryTreeNode.
        """
        #First, generate proximity nodes to the graphs associated with the leaves from the query tree
        self.generate_proximity_nodes_in_all_leaf_graphs()
        
        #Then, generate proximity nodes to the graphs associated with the rest of the nodes in the tree
        self.query_tree.operate_non_leaf_graphs_from_leaves()
        
        #Finally, generate frequency criteria VicinityNodes for the root BinaryTreeNode.
        self.generate_frequency_criteria_nodes_for_root()
    
    
    def generate_frequency_criteria_nodes_for_root(self) -> None:
        """
        Generate frequency criteria VicinityNodes for the root BinaryTreeNode.\n
        This function creates a frequency dictionary of all the sentence words, to then multiply
        its frequencies by the document weight and finally create frequency criteria VicinityNodes 
        for the root BinaryTreeNode and adds them to the graph.
        """
        root_graph = self.query_tree.get_graph()
        if not root_graph:
            return
        
        terms_freq_dict = self.get_terms_frequency_dict()
        graph_terms_list = root_graph.get_terms_from_all_nodes()
        query_terms_list = self.query_tree.get_query_terms_str_list_with_underscores()
        query_terms_list_without_underscores = VectorUtils.split_and_extend_from_underscore_values(query_terms_list)
        
        for term, frequency in terms_freq_dict.items():
            if term in query_terms_list_without_underscores:    # Validate that the term is not a query term
                continue
            
            freq_score: float = frequency * self.__weight
            
            if term in graph_terms_list:    # If the term is in the graph, modify its frequency score
                graph_node = root_graph.get_node_by_term(term)
                if graph_node:  # Double check
                    graph_node.frequency_score = freq_score
            else:   # If the term isn't in the graph, add a new frequency node
                freq_criteria_node = VicinityNode(
                    term=term,
                    frequency_score = freq_score,
                    proximity_score=0.0,
                    criteria="frequency"
                )
                root_graph.add_node(freq_criteria_node)


    def get_terms_frequency_dict(self) -> dict[str, int]:
        """
        This method calculates the frequency of terms from the preprocessed sentence

        Returns
        -------
        terms_freq_dict : dict[str, int]
            A dictionary containing the frequency of terms.
        """
        # Validate if the preprocessed sentence is not empty
        if not self.__preprocessed_text:
            return dict()
        
        # Normalize the text by converting it to lowercase and removing punctuation
        words = re.findall(r'\b\w+\b', self.__preprocessed_text.lower())
        
        # Count the frequency of each word using Counter
        frequencies = Counter(words)
        
        return dict(frequencies)


    def generate_proximity_nodes_in_all_leaf_graphs(self):
        """
        Generate proximity nodes to the graphs associated with the leaves from the query tree.\n
        This function iterates over all the leaf nodes in the query tree and calls the 
        `generate_proximity_nodes_in_leaf_graph` method for each leaf node. This method is responsible 
        for generating proximity nodes to the graph associated with the leaf node.
        """
        for leaf_node in self.query_tree.get_query_terms_as_leaves():
            self.generate_proximity_nodes_in_leaf_graph(leaf_node)
    

    def get_term_positions_dict(self) -> defaultdict[str, list[int]]:
        """
        Calculate a dictionary to store the list of positions for each text term.

        Returns
        -------
        term_positions_dict : defaultdict[str, list[int]]
            A dictionary with a list of positions for each term of the text
        """
        term_positions_dict = defaultdict(list)
        # Tokenize and normalize the text
        terms = re.findall(r'\w+', self.__preprocessed_text.lower())
        for i, term in enumerate(terms):
            term_positions_dict[term].append(i)
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
    

    def generate_proximity_nodes_in_leaf_graph(self, 
            leaf_node: BinaryTreeNode,
            ) -> None:
        """
        Generate proximity nodes for the graph associated with the sentence, based on the isolated query 
        term of the leaf from the query tree. This method calculates the proximity scores of the terms 
        associated with the query term of the graph, and adds each of them to their respective node.

        Parameters
        ----------
        leaf_node : BinaryTreeNode
            A leaf node associated with the sentence tree
        """
        query_term: str = leaf_node.value
        terms_prox_freq_dict: dict[str, int] = self.get_terms_proximity_frequency_dict(query_term)

        # Iterate over each term in the proximity frequency dictionary
        for neighbour_term in terms_prox_freq_dict.keys():
            # Retrieve the list of frequencies by distance for the term
            # E.g.   {device: {iot: [1, 0, 2, 0]}}  ->  [1, 0, 2, 0]
            freq_by_distance_list: list[int] = self.__vicinity_matrix.get(neighbour_term).get(query_term)

            # Construct a list of occurrence of distances
            # E.g.  [1, 0, 2, 0] -> [1, 3, 3]
            distance_occurrence_list: list[int] = []
            for idx, frequency in enumerate(freq_by_distance_list):
                distance_occurrence_list.extend([idx+1] * frequency)
            
            # Transform the values ​​in the list to the following formula: 1 / (2 ^ (distance - 1) )
            # E.g.  [1, 3, 3] -> [1, 0.25, 0.25]
            distance_score_calculation_list = VectorUtils.calculate_distance_score_list(distance_occurrence_list)
            
            # Calculate the proximity score of the term, multiplying the score list sum by the document weight
            # E.g.  sum([1, 0.25, 0.25]) * doc_weight  ->  1.5 * doc_weight
            score_list_sum: float = sum(distance_score_calculation_list)
            prox_score = score_list_sum * self.__weight

            # Round the values to 6 decimal places for better readability
            prox_score = round(prox_score, 6)
            
            # Initialize the new vicinity node and add it to the leaf node graph
            new_node = VicinityNode(term=neighbour_term, proximity_score=prox_score, 
                                    frequency_score=0.0, criteria="proximity")
            
            leaf_node.graph.add_node(new_node)

    

    def get_terms_proximity_frequency_dict(self, 
            query_term: str
            ) -> dict[str, int]:
        """
        This method calculates the proximity frequency of terms from the vicinity matrix 
        of the sentence, by a specified query term

        Parameters
        ----------
        query_term : str
            A query term of the graph

        Returns
        -------
        terms_prox_freq_dict : dict[str, int]
            A dictionary containing the proximity frequency of terms. Each dictionary has keys corresponding 
            to terms, and values represent the frequency of the term within the sentence.

        """
        terms_prox_freq_dict = {}

        # Iterate over each term and its frequencies in the document
        for neighbor_term, distance_freq_by_query_term in self.__vicinity_matrix.items():
            # Checks if the query word is in the distance-frequency 
            # dictionary keys and if the neighbor term isn't a query term
            if (query_term in distance_freq_by_query_term):
                sum_of_freqs_in_query_term = sum(distance_freq_by_query_term[query_term])
                if sum_of_freqs_in_query_term > 0:   # Double check
                    terms_prox_freq_dict[neighbor_term] = sum_of_freqs_in_query_term
            
        return terms_prox_freq_dict
    

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
            if (query_term_with_spaces in transformed_sentence_str) and ('_' in query_term_with_underscores): 
                sentence_str_with_underscores_in_query_terms = transformed_sentence_str.replace(query_term_with_spaces, 
                                                                                                query_term_with_underscores)
        return sentence_str_with_underscores_in_query_terms



class Document(QueryTreeHandler):
    
    def __init__(self, query: BinaryExpressionTree, abstract: str = "", title: str = "", 
                 doc_id: str = "1", weight: float = 1.0, ranking_position: int = 1):
        """
        Initialize a new Document object.

        Parameters:
        ranking (Ranking): The ranking associated with this document.
        query (BinaryExpressionTree): The query tree for the document.
        abstract (str, optional): The abstract of the document. Default is an empty string.
        title (str, optional): The title of the document. Default is an empty string.
        doc_id (str, optional): The unique identifier of the document. Default is "1".
        weight (float, optional): The weight of the document in the ranking. Default is 1.0.
        ranking_position (int, optional): The position of the document in the ranking. Default is 1.
        """
        super().__init__(query_tree=query)
        self.__title = title
        self.__abstract = abstract
        self.__preprocessed_text: str = ""
        self.__doc_id = doc_id
        self.__weight = weight
        self.__ranking_position = ranking_position
        self.__sentences: list[Sentence] = []

        self.__calculate_sentences_list_from_documents()
    

    @property
    def abstract(self) -> str:
        """
        Retrieve the abstract of the current Document object.

        Returns:
        str: The abstract of the document.
        """
        return self.__abstract
    

    @property
    def title(self) -> str:
        """
        Retrieve the title of the current Document object.

        Returns:
        str: The title of the document.
        """
        return self.__title
    

    @property
    def doc_id(self) -> str:
        """
        Retrieve the unique identifier of the current document.

        Returns:
        str: The unique identifier of the document.
        """
        return self.__doc_id
    

    @property
    def sentences(self) -> list[Sentence]:
        """
        Retrieve a list of Sentence objects from the current Document object.

        Returns:
        list[Sentence]: A list of Sentence objects representing the sentences in the document.
        """
        return self.__sentences
    

    @property
    def weight(self) -> float:
        """
        Retrieve the weight of the current document, obtained by its position in the ranking.

        Returns:
        float: The weight of the document.
        """
        return self.__weight


    @property
    def ranking_position(self) -> int:
        """
        Retrieve the ranking position of the current document.

        Parameters:
        None

        Returns:
        int: The ranking position of the document. The position index is 1-based.
        """
        return self.__ranking_position
    
    
    @property
    def preprocessed_text(self) -> str:
        """
        Retrieve the preprocessed text of the document, which consists of the title and abstract joined.
        
        Parameters:
        None

        Returns:
        str: The preprocessed text of the document
        """
        return self.__preprocessed_text
    

    def get_sentence_by_raw_text(self, text: str) -> Sentence | None:
        """
        Retrieve a Sentence object from the current Document object based on its raw text.

        Parameters:
        text (str): The raw text of the sentence to be retrieved.

        Returns:
        Sentence | None: The Sentence object with the specified raw text, or None if no such sentence is found.
        """
        for sentence in self.__sentences:
            if sentence.raw_text == text:
                return sentence
        print(f'No sentence with raw text: {text}')
        return None
    

    def get_sentence_by_position_in_doc(self, position: int) -> Sentence | None:
        """
        Retrieve a Sentence object from the current Document object based on its position in the document.

        Parameters:
        position (int): The position of the sentence to be retrieved. The position index is 0-based.

        Returns:
        Sentence | None: The Sentence object at the specified position, or None if no such sentence is found.
        """
        try:
            return self.__sentences[position]
        except IndexError:
            print(f'No sentence with position: {position}')
            return None
    

    def get_list_of_query_trees_from_sentences(self) -> list[BinaryExpressionTree]:
        """
        Retrieves a list of query trees from the sentences of the current document.

        Returns:
        list[BinaryExpressionTree]: A list of BinaryExpressionTree objects representing 
        the query trees of the sentences in the document.
        """
        list_of_query_trees = [sentence.query_tree for sentence in self.__sentences]
        return list_of_query_trees
    

    def do_document_and_sentences_text_transformations(self,
            stop_words: tuple[str] = (), 
            lema: bool = True,
            stem: bool = False
            ) -> None:
        """
        Apply some text transformations to each sentence of the document and get the 
        document preprocessed text from the sentences preprocessed text.

        Parameters
        ----------
        stop_words : tuple[str]
            List of stop words to be removed from the sentence
        lema : bool
            If True, lematization is applied
        stem : bool
            If True, stemming is applied
        """
        # Document query graphs is re-initialized
        self.query_tree.remove_graphs_for_each_node()
        
        # Do text transformations to each sentence
        for sentence in self.__sentences:
            sentence.do_text_transformations(stop_words, lema, stem)
            
        # Get the document preprocessed text from the sentences preprocessed text
        preprocessed_text: str = ""
        for index, sentence in enumerate(self.__sentences):
            preprocessed_text += sentence.preprocessed_text
            if index < len(self.__sentences) - 1:
                preprocessed_text += " "
        
        self.__preprocessed_text = preprocessed_text


    def generate_graph_nodes_of_doc_and_sentences(self) -> None:
        """
        Generate all the nodes associated with the document graphs, along with 
        the graphs of their sentences, based on their query trees.
        """
        #Generate graph nodes in sentences
        for sentence in self.__sentences:
            sentence.generate_nodes_in_tree_graphs()
        
        #Generate graph nodes in the current document
        union_of_trees = self.__get_union_of_sentences_trees()
        if union_of_trees:
            self.query_tree = union_of_trees
    

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
            query_copy = copy.deepcopy(self.query_tree)
            sentence_obj = Sentence(raw_text=sentence_str, query=query_copy, 
                                    position_in_doc=index, weight=self.__weight)
            self.__sentences.append(sentence_obj)

    
    def __get_list_of_sentence_strings(self) -> list[str]:
        """
        Transform the text of the current document (title + abstract) into a list of string 
        sentences. Split the text by dots and trim the text portions.

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
            abstract_list_of_sentence_str = [sentence.strip() for sentence in abstract_list_of_sentence_str]
            list_of_sentence_str.extend(abstract_list_of_sentence_str)
        return list_of_sentence_str
    

    def __get_union_of_sentences_trees(self) -> BinaryExpressionTree | None:
        """
        Get the union between the list of query trees associated with the sentences of the document.

        Returns
        -------
        union_of_trees : BinaryExpressionTree
            The union between the sentence query trees.
        """
        query_trees_list = self.get_list_of_query_trees_from_sentences()
        union_of_trees = self.query_tree.get_union_of_trees(query_trees_list)
        return union_of_trees



class TextTransformationsConfig:
    def __init__(self, stop_words: tuple[str] = (), lemmatization: bool = True, stemming: bool = False):
        """
        Initialize a new TextTransformationsConfig object.

        Parameters:
        stop_words (tuple[str], optional): A tuple of stop words to be excluded from the text transformations. Default is an empty tuple.
        lemmatization (bool, optional): A boolean indicating whether lemmatization should be applied to the text transformations. Default is True.
        stemming (bool, optional): A boolean indicating whether stemming should be applied to the text transformations. Default is False.
        """
        self.__stop_words = stop_words
        self.__lemmatization = lemmatization
        self.__stemming = stemming


    def get_transformations_params(self) -> tuple[tuple[str], bool, bool]:
        """
        Returns the stop words, lemmatization flag, and stemming flag used for text transformations.

        Returns:
        tuple[tuple[str], bool, bool]: A tuple containing the stop words, lemmatization flag, and stemming flag.
        """
        return self.__stop_words, self.__lemmatization, self.__stemming



class Ranking(QueryTreeHandler):
    
    def __init__(self, query_text: str, nr_search_results: int = 10, ranking_weight_type: str = 'linear', 
                 stop_words: tuple[str] = (), lemmatization: bool = True, stemming: bool = False):
        """
        Initialize a new Ranking object.
        
        Parameters:
        query_text (str): The query text for the ranking.
        nr_search_results (int, optional): The number of search results to retrieve from the ranking. Default is 10.
        ranking_weight_type (str, optional): The type of weighting to be applied to the ranking. It can be 'none', 'linear' or 'inverse'. Default is 'linear'.
        stop_words (tuple[str], optional): A tuple of stop words to be excluded from the ranking. Default is an empty tuple.
        lemmatization (bool, optional): A boolean indicating whether lemmatization should be applied to the ranking. Default is True.
        stemming (bool, optional): A boolean indicating indicating whether stemming should be applied to the ranking. Default is False.
        """
        self.__initialize_binary_expression_tree(query_text)
        self.__nr_search_results = nr_search_results
        self.__ranking_weight_type = ranking_weight_type
        self.__text_transformations_config = TextTransformationsConfig(stop_words, lemmatization, stemming)
        self.__documents: list[Document] = []

        self.__do_text_transformations_to_query_terms()


    @property
    def documents(self) -> list[Document]:
        """
        Retrieve a list of documents from the ranking.

        Returns:
        list[Document]: A list of Document objects representing the documents in the ranking.
        """
        return self.__documents 
    

    @property
    def text_transformations_config(self) -> TextTransformationsConfig:
        """
        Retrieve the text transformations configuration used in the ranking.

        Returns:
        TextTransformationsConfig: The text transformations configuration used in the ranking.
        """
        return self.__text_transformations_config
    

    def get_document_by_title(self, title: str) -> Document | None:
        """
        Retrieve a document from the ranking based on its title.

        Parameters:
        title (str): The title of the document to be retrieved.

        Returns:
        Document | None: The document with the specified title, or None if the title is not found.
        """
        for document in self.__documents:
            if document.title == title:
                return document
        print(f'No document with title: {title}')
        return None
    

    def get_document_by_id(self, id: str) -> Document | None:
        """
        Retrieve a document from the ranking based on its id.

        Parameters:
        id (str): The id of the document to be retrieved.

        Returns:
        Document | None: The document with the specified id, or None if the id is not found.
        """
        for document in self.__documents:
            if document.doc_id == id:
                return document
        print(f'No document with id: {id}')
        return None
    

    def get_document_by_ranking_position(self, position: int) -> Document | None:
        """
        Retrieve a document from the ranking based on its ranking position.

        Parameters:
        position (int): The position of the document in the ranking.

        Returns:
        Document | None: The document at the specified ranking position, or None if the position is invalid.
        """
        try:
            return self.__documents[position-1]
        except IndexError:
            print(f'No document with ranking position: {position}')
            return None
    

    def get_list_of_query_trees_from_documents(self) -> list[BinaryExpressionTree]:
        """
        Returns a list of query trees from the documents of the current ranking.

        Returns:
        list[BinaryExpressionTree]: A list of BinaryExpressionTree objects representing 
        the query trees of the documents in the ranking.
        """
        list_of_query_trees = [document.query_tree for document in self.__documents]
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
        articles_dicts : list[dict]
            A list of dictionaries containing the attributes 'title', 'abstract' and 'article_number'.
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
        if not self.__documents:
            print("No documents were found")
        else:
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

        Parameters:
        nr_of_graph_terms (int, optional): Configured number of terms in the graph. Default is 5.
        limit_distance (int, optional): Maximal distance of terms used to calculate the vicinity. Default is 4.
        include_query_terms (bool, optional): If True, the query term is included in the vicinity. Default is True.
        summarize (str, optional): Summarization type to operate distances in the vicinity matrix for each sentence. 
                                    It can be 'mean' or 'median'. Default is 'mean'.
        """
        parameters_tuple = (nr_of_graph_terms, limit_distance, include_query_terms, summarize)

        #Initialize the graphs of the query tree associated with the ranking
        self.query_tree.initialize_graph_for_each_node(*parameters_tuple)

        #Initialize the graphs of the query trees associated with the documents of the ranking
        for document in self.__documents:
            document.query_tree.initialize_graph_for_each_node(*parameters_tuple)
        
        #Initialize the graphs of the query trees associated with the sentences from the documents of the ranking
        for document in self.__documents:
            for sentence in document.sentences:
                sentence.query_tree.initialize_graph_for_each_node(*parameters_tuple)
    

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
        union_of_trees = self.__get_union_of_documents_trees()
        if union_of_trees:
            self.query_tree = union_of_trees
    

    def __get_union_of_documents_trees(self) -> BinaryExpressionTree | None:
        """
        Get the union between the list of query trees associated with the 
        documents of the ranking.

        Returns
        -------
        union_of_trees : BinaryExpressionTree
            The union between the document query trees.
        """
        query_trees_list = self.get_list_of_query_trees_from_documents()
        union_of_trees = self.query_tree.get_union_of_trees(query_trees_list)
        return union_of_trees


    def __do_text_transformations_to_query_terms(self) -> None:
        """
        Apply text transformations to the query terms of the ranking. Lower the text, tokenize, remove 
        punctuation, stopwords, finally do stemming and lemmatization if specified.
        """
        self.query_tree.do_text_transformations_to_query_terms(*self.__text_transformations_config.get_transformations_params())


    def __get_ieee_xplore_ranking(self) -> list[dict]:
        """
        Get a ranking of articles from IEEE-Xplore.

        Returns
        -------
        results : list[dict]
            A list of dictionaries, where each dictionary represents an article. Each dictionary 
            contains the article's attributes such as 'title', 'abstract', and 'article_number'.
        """
        xplore_id = '6g7w4kfgteeqvy2jur3ak9mn'
        query = XPLORE(xplore_id)
        query.outputDataFormat='object'
        query.maximumResults(self.__nr_search_results)
        query.queryText(self.query_tree.raw_query)
        data = query.callAPI()
        results = data.get('articles', [{}])
        
        return results


    def __calculate_ranking_as_weighted_documents_and_do_text_transformations(self,
            articles_dicts: list[dict],
            ) -> None:
        """
        Transforms the list of article dictionaries into a list of Document type objects, which have a weight
        associated with their position in the received list. Objects will be appended to the ranking.
        In addition, text pre-processing of documents is carried out.
        If weighted <> none : the document will be weighted depending on its position in the ranking.  

        Parameters
        ----------
        articles_dicts : list[dict]
            List of article dictionaries to be processed as Document type objects and to be part of the ranking
        """
        # Ranking documents and ranking query graphs are re-initialized
        self.__documents = []
        self.query_tree.remove_graphs_for_each_node()

        for index, article in enumerate(articles_dicts):
            try:
                new_doc = self.__get_new_document_by_article(article, index, results_size=len(articles_dicts))
                new_doc.do_document_and_sentences_text_transformations(*self.__text_transformations_config.get_transformations_params())
                self.__documents.append(new_doc)
            except Exception as e:  # If a document has errors, ignore it and continue to the next document
                print(f"An error occurred while processing article at index {index}: {e}")
                continue
    

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
            Article dictionary with title, abstract and id.
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

        # Check if abstract, title and doc_id are strings
        if not all(isinstance(atribute, str) for atribute in [_abstract, _title, _doc_id]):
            raise TypeError("All variables must be strings")
        
        # Check if abstract and title are not empty strings
        if not _abstract and not _title:
            raise ValueError("Abstract and title cannot be both empty")

        _ranking_pos = index + 1
        _weight = MathUtils.calculate_document_weight(results_size, _ranking_pos, self.__ranking_weight_type)
        _query_copy = copy.deepcopy(self.query_tree)
        new_doc = Document(query=_query_copy, abstract=_abstract, title=_title, doc_id=_doc_id, 
                            weight=_weight, ranking_position=_ranking_pos)
        
        return new_doc
    
    
    def __initialize_binary_expression_tree(self, 
        query_text: str
        ) -> None:
        """
        Validate the query text and initialize the binary expression tree.

        Parameters:
        query_text (str): The input query text to be used to create the binary expression tree.

        Raises:
        ValueError: If the query text has invalid syntax, a ValueError is raised with a descriptive message.
        """
        self.__validate_alnum_query_text_not_empty(query_text)
        try:
            super().__init__(query_tree=BinaryExpressionTree(query_text))
        except Exception as e:
            raise ValueError("Invalid query syntax: " + repr(e))
    
    
    def __validate_alnum_query_text_not_empty(self, 
        query_text: str
        ) -> None:
        """
        Validate that the alphanumeric version of the query text is not empty, otherwise raise an exception.

        Parameters:
        query_text (str): The input query text to be validated.

        Returns:
        None: This function does not return any value. It raises an exception if the query text is empty.

        Raises:
        ValueError: If the alphanumeric version of the query text is empty.
        """
        alnum_query_text = re.sub(r'\W+', '', query_text)
        if alnum_query_text == '':
            raise ValueError("Query text cannot be empty")