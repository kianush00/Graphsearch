import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import copy
import math

from models.ieee_xplore.xploreapi import XPLORE
from models.srex.vicinity_graph import VicinityGraph
from models.srex.vicinity_graph import VicinityNode
from models.srex.binary_expression_tree import BinaryExpressionTree
from models.srex.binary_expression_tree import BinaryTreeNode
from utils.text_utils import TextUtils




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
        self.__vicinity_matrix: dict[str, dict[str, list[float]]] = {}


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
            stem: bool = False
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
                        limit_distance = self.get_graph().get_config().get_limit_distance()
                        freq_neighborhood_positions = self.calculate_distances_between_term_positions(query_positions, 
                                                                                                       term_positions,
                                                                                                       limit_distance)

                        if (any(frq > 0 for frq in freq_neighborhood_positions)):
                            if (first_one):
                                vicinity_matrix[term] = {}
                                first_one = False
                            vicinity_matrix[term][query_term] = freq_neighborhood_positions

        self.__vicinity_matrix = vicinity_matrix
    

    def generate_nodes_in_tree_graphs(self) -> None:
        """
        Generate nodes to all the graphs associated with the sentence tree, based 
        on their isolated query terms as leaves from the query tree.
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


    def calculate_distances_between_term_positions(self,
            term1_positions: list[int], 
            term2_positions: list[int],
            limit_distance: int
            ) -> list[float]:
        """
        Compare the positions vectors of two terms, and return the list of 
        frequencies per distance between query terms and vicinity terms
        
        E.g.
        term1_positions = [0, 2, 4, 6]
        term2_positions = [1, 3, 5, 7]
        limit_distance = 7
        result = [7.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0]

        Parameters
        ----------
        term1_positions : list[int]
            List of positions of the first term
        term2_positions : list[int]
            List of positions of the second term
        limit_distance : int
            Limit distance to calculate between the two term positions lists

        Returns
        -------
        frequencies_per_distance : list[float]
            List of frequencies per distance between query terms and vicinity terms
        """
        frequencies_per_distance = [0] * limit_distance

        for term1_pos in term1_positions:
            for term2_pos in term2_positions:
                absolute_distance = abs(term1_pos-term2_pos)
                if (absolute_distance <= limit_distance):
                    frequencies_per_distance[absolute_distance-1] += 1

        return frequencies_per_distance
    

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
        terms_freq_dict = self.get_terms_frequency_dict(query_term)

        # Iterate over each term in the frequency dictionary
        for neighbor_term in terms_freq_dict.keys():
            # Retrieve the list of frequencies by distance for the term
            list_of_freq_by_distance = self.__vicinity_matrix.get(neighbor_term).get(query_term)
            distance_calculation_list = []

            # Construct a list of distances multiplied by its frequencies
            # E.g.  [1, 0, 2, 0] -> [1, 3, 3]
            for idx, frequency in enumerate(list_of_freq_by_distance):
                distance_calculation_list.extend([idx+1] * frequency)
            
            #Summarize the list of distances based on the graph settings (it can be: mean or median)
            if self.get_graph().get_config().get_summarize() == 'median':
                _distance = np.median(distance_calculation_list)
            else:
                _distance = np.mean(distance_calculation_list)
            
            # Calculate the ponderation of the term, by the formula:  p = (1 + log(tf)) * w
            term_frequency = terms_freq_dict.get(neighbor_term)
            _ponderation = (1 + math.log10(term_frequency)) * self.__weight

            # Round the values to 6 decimal places for better readability
            _distance = round(_distance, 6)
            _ponderation = round(_ponderation, 6)
                
            # Calculate the mean distance for the term
            new_node = VicinityNode(term=neighbor_term, ponderation=_ponderation, distance=_distance)
            leaf_node.graph.add_node(new_node)

    

    def get_terms_frequency_dict(self, 
            query_term: str
            ) -> dict[str, float]:
        """
        This method calculates the frequency of terms from the vicinity matrix 
        of the sentence, by a specified query term

        Parameters
        ----------
        query_term : str
            The query term of the graph

        Returns
        -------
        terms_freq_dict : dict[str, float]
            A dictionary containing the frequency of terms. Each dictionary has keys corresponding 
            to terms, and values represent the frequency of the term within the sentence.

        """
        terms_freq_dict = {}

        # Iterate over each term and its frequencies in the document
        for neighbor_term, distance_freq_by_query_term in self.__vicinity_matrix.items():
            # Checks if the query word is in the distance-frequency 
            # dictionary keys and if the neighbor term isn't a query term
            if (query_term in distance_freq_by_query_term):
                sum_of_freqs_in_query_term = sum(distance_freq_by_query_term[query_term])
                if sum_of_freqs_in_query_term > 0:
                    terms_freq_dict[neighbor_term] = sum_of_freqs_in_query_term
            
        return terms_freq_dict
    

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
            sentence.generate_nodes_in_tree_graphs()
        
        #Generate graph nodes in the current document
        union_of_trees = self.__get_union_of_sentences_trees()
        if union_of_trees:
            self.set_query_tree(union_of_trees)
    

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
    

    def __get_union_of_sentences_trees(self) -> BinaryExpressionTree | None:
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
        union_of_trees = self.__get_union_of_documents_trees()
        if union_of_trees:
            self.set_query_tree(union_of_trees)
    

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
            try:
                new_doc = self.__get_new_document_by_article(article, index, results_size=len(articles_dicts))
                new_doc.do_text_transformations_by_sentence(*self.__text_transformations_config.get_transformations_params())
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

        # Check if abstract, title and doc_id are strings
        if not all(isinstance(atribute, str) for atribute in [_abstract, _title, _doc_id]):
            raise TypeError("All variables must be strings")
        
        # Check if abstract and title are not empty strings
        if not _abstract and not _title:
            raise ValueError("Abstract and title cannot be both empty")

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
            factor = float((results_size - ((ranking_position - 1) * 0.7)) / results_size)
        elif (weighted=='inverse'):
            factor = float(1 / (((ranking_position - 1) * 0.05) + 1))
        else:
            factor = 1.0

        return factor