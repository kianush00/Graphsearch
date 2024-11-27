import re
import copy
from graphviz import Graph
from utils.vector_utils import VectorUtils




class VicinityGraphConfig:
    def __init__(self, nr_of_graph_terms: int = 5, limit_distance: int = 4, 
                 include_query_terms: bool = True, summarize: str = 'mean'):
        """
        Initialize a VicinityGraph object with the specified parameters.

        Parameters:
        nr_of_graph_terms (int): The maximum number of terms to include in the graph. Default is 5.
        limit_distance (int): The maximum distance allowed for terms in the graph. Default is 4.
        include_query_terms (bool): Whether to include terms from the query in the graph. Default is True.
        summarize (str): The method to summarize ponderations when calculating cosine similarity. Default is 'mean'.
        """
        self.__number_of_graph_terms = nr_of_graph_terms
        self.__limit_distance = limit_distance
        self.__include_query_terms = include_query_terms
        self.__summarize = summarize    # it can be: 'mean' or 'median'
    

    def get_number_of_graph_terms(self) -> int:
        """
        Returns the number of graph terms to be displayed.

        Returns:
        int: The number of graph terms.
        """
        return self.__number_of_graph_terms


    def get_limit_distance(self) -> int:
        """
        Returns the limit distance for the vicinity nodes.

        Returns:
        int: The limit distance.
        """
        return self.__limit_distance


    def get_include_query_terms(self) -> bool:
        """
        Returns whether the query terms should be included in the graph.

        Returns:
        bool: True if query terms should be included, False otherwise.
        """
        return self.__include_query_terms


    def get_summarize(self) -> str:
        """
        Returns the method used to summarize the graph.

        Returns:
        str: The method used for summarizing the graph.
        """
        return self.__summarize


    def get_config_params(self) -> tuple[int, int, bool, str]:
        """
        Returns the configuration parameters of the graph.

        Returns:
        tuple: A tuple containing the number of graph terms, limit distance, 
            include query terms, and summarization method.
        """
        return self.__number_of_graph_terms, self.__limit_distance, self.__include_query_terms, self.__summarize



class VicinityNode:
    
    def __init__(self, term: str, frequency_score: float = 0.0, proximity_score: float = 0.0, 
                 criteria: str = "frequency"):
        """
        Initialize a new instance of the VicinityNode class.

        Parameters:
        term (str): The term of the node.
        frequency_score (float, optional): The frequency score of the node. Default is 0.0.
        proximity_score (float, optional): The proximity score of the node. Default is 0.0.
        criteria (str, optional): The criteria of the node. It can be: 'proximity', 'frequency' or 
        'exclusion'. Default is 'proximity'.
        """
        self.__term = term
        self.__frequency_score = frequency_score
        self.__proximity_score = proximity_score
        self.set_criteria(criteria)
    

    def get_term(self) -> str:
        """
        Returns the term of the current node.

        Returns:
        str: The term of the current node.
        """
        return self.__term


    def get_frequency_score(self) -> float:
        """
        Returns the frequency score of the current node.

        Returns:
        float: The frequency score of the current node.
        """
        return self.__frequency_score


    def set_frequency_score(self, frequency_score: float) -> None:
        """
        Sets the frequency score of the current node.

        Parameters:
        frequency_score (float): The new frequency score value.
        """
        self.__frequency_score = frequency_score
    
    
    def get_proximity_score(self) -> float:
        """
        Returns the proximity score of the current node.

        Returns:
        float: The proximity score of the current node.
        """
        return self.__proximity_score


    def set_proximity_score(self, proximity_score: float) -> None:
        """
        Sets the proximity score of the current node.

        Parameters:
        proximity_score (float): The new proximity score value.
        """
        self.__proximity_score = proximity_score
    
    
    def get_criteria(self) -> str:
        """
        Returns the criteria used to determine the type of node calculation (it can be: proximity, frequency or exclusion).
        - 'proximity': The system calculates the proximity of the term to the query in the document, to get the new ranking.
        - 'frequency': The system calculates the frequency of the term in the document to get the new ranking.
        - 'exclusion': The system excludes the term from the ranking, adding a NOT operator in the query string and 
        making a new query with the term excluded.
        
        Returns:
        str: The criteria used to determine proximity.
        """
        return self.__criteria
    
    
    def set_criteria(self, criteria: str) -> None:
        """
        Sets the criteria used to determine the type of node calculation.
        
        Parameters:
        criteria (str): The new criteria value.
        """
        if criteria not in ['proximity', 'frequency', 'exclusion']:
            raise ValueError("Invalid criteria. Criteria can be: proximity, frequency or exclusion.")
        if criteria == 'proximity' and self.__proximity_score <= 0.0:
            raise ValueError("Proximity score cannot be negative or zero when criteria is 'proximity'.")
        
        self.__criteria = criteria


    def __str__(self) -> str:
        """
        Returns a string representation of the current node.

        Returns:
        str: A string representation of the current node.
        """
        term = self.__term
        freq_score = self.__frequency_score
        prox_score = self.__proximity_score
        criteria = self.__criteria
        string = f"TERM: {term} ; FREQUENCY_SCORE: {freq_score} ; PROXIMITY_SCORE: {prox_score} ; CRITERIA: {criteria}"
        return string



class VicinityGraph:
        
    def __init__(self, subquery: str, nr_of_graph_terms: int = 5, limit_distance: int = 4, 
                 include_query_terms: bool = True, summarize: str = 'mean'):
        """
        Initialize a VicinityGraph object.

        Parameters:
        subquery (str): The subquery for which the graph is being created.
        nr_of_graph_terms (int, optional): The maximum number of terms to include in the graph. Defaults to 5.
        limit_distance (int, optional): The maximum distance for terms to be included in the graph. Defaults to 4.
        include_query_terms (bool, optional): Whether to include terms from the subquery in the graph. Defaults to True.
        summarize (str, optional): The method to summarize ponderations when calculating cosine similarity. Defaults to 'mean'.
        """
        self.subquery = subquery
        self.__config: VicinityGraphConfig = VicinityGraphConfig(nr_of_graph_terms, limit_distance, 
                                                                 include_query_terms, summarize)
        self.__nodes: list[VicinityNode] = []


    def get_config(self) -> VicinityGraphConfig:
        """
        Returns the configuration object of the VicinityGraph.

        Returns:
        VicinityGraphConfig: The configuration object of the VicinityGraph.
        """
        return self.__config
    

    def get_all_nodes_sorted(self) -> list[VicinityNode]:
        """
        Return the list of all nodes (all criteria types) sorted by proximity score in descending order.

        Returns:
        list[VicinityNode]: A list of VicinityNode objects sorted by proximity score in descending order.
        """
        sorted_nodes = sorted(self.__nodes, key=lambda node: node.get_proximity_score(), reverse=True)
        return sorted_nodes
    
    
    def get_proximity_nodes_sorted(self) -> list[VicinityNode]:
        """
        Return the list of proximity nodes sorted by proximity score in descending order.

        Returns:
        list[VicinityNode]: A list of proximity VicinityNode objects sorted by proximity score in descending order.
        """
        sorted_proximity_nodes = [node for node in self.__nodes if node.get_criteria() == "proximity"]
        sorted_proximity_nodes = sorted(sorted_proximity_nodes, key=lambda node: node.get_proximity_score(), reverse=True)
        return sorted_proximity_nodes
    

    def get_node_by_term(self, term: str) -> VicinityNode | None:
        """
        Retrieve a node from the graph by its term.

        Parameters:
        term (str): The term of the node to be retrieved.

        Returns:
        VicinityNode | None: The node with the given term if found, otherwise None.
        If a node with the given term is not found, it prints "No node with term".
        """
        for node in self.__nodes:
            if node.get_term() == term:
                return node
        print(f"No node with term {term}")
        return None
    

    def get_terms_str_from_all_nodes(self) -> list[str]:
        """
        Retrieves a list of terms from all the nodes of the current graph.

        Returns:
        list[str]: A list of terms from all the nodes of the current graph.
        """
        node_terms = [node.get_term() for node in self.get_all_nodes_sorted()]
        return node_terms
    
    
    def get_terms_from_proximity_nodes(self) -> list[str]:
        """
        Retrieves a list of terms from the proximity nodes of the current graph.

        Returns:
        list[str]: A list of terms from the proximity nodes of the current graph.
        """
        node_terms = [node.get_term() for node in self.get_proximity_nodes_sorted()]
        return node_terms


    def add_node(self, node: VicinityNode) -> None:
        """
        Adds a node to the graph.

        Parameters:
        node (VicinityNode): The node to be added to the graph. This node should be an instance of the VicinityNode class.
        """
        self.__nodes.append(node)


    def delete_node_by_term(self, term: str) -> None:
        """
        Deletes a node from the graph based on the given term. \n
        The function iterates through the nodes in the graph and removes the node with the given term.
        If no node with the given term is found, it prints "No node with term".

        Parameters:
        term (str): The term of the node to be deleted.
        """
        for node in self.__nodes:
            if node.get_term() == term:
                self.__nodes.remove(node)
                return
        print(f"No node with term {term}")
    

    def __str__(self) -> str:
        """
        Returns a string representation of the VicinityGraph object.

        The string representation includes the subquery and the details of each node in the graph.
        Each node is represented as a string in the format: 
        "term ; frequency score ; proximity score ; criteria"

        Returns:
        string (str): A string representation of the VicinityGraph object.
        """
        string = "SUBQUERY: " + self.subquery
        for node in self.get_all_nodes_sorted():
            string += "\n" + str(node)
        return string
    

    def get_graph_as_dict(self) -> dict[str, dict[str, float | str]]:
        """
        Returns the graph as a dictionary.

        e.g.   {'v_term1': {'frequency_score': 3.0, 'proximity_score': 3.4, 'criteria': 'proximity'},\n 
            'v_term2': {'frequency_score': 2.0, 'proximity_score': 0.0, 'criteria': 'frequency'},\n
            ...}

        Returns:
        dict[str, dict[str, float]]: A dictionary representing the graph. The keys 
        are the terms, and the values are dictionaries containing the 'frequency_score', 
        'proximity_score' and 'criteria' of each term.
        """
        graph_dict = {n.get_term(): {'frequency_score': n.get_frequency_score(),
                                     'proximity_score': n.get_proximity_score(),
                                     'criteria': n.get_criteria()} for n in self.get_all_nodes_sorted()}
        return graph_dict

    
    def get_viewable_graph_copy(self) -> 'VicinityGraph':
        """
        Returns a copy of the graph, with the vicinity terms with the highest proximity score 
        limited by the number of graph terms configured in the current graph, and 
        ignores the vicinity nodes that include the query terms.
        
        Returns:
        graph_copy (VicinityGraph): A copy of the graph with the specified conditions applied.
        """
        graph_copy = VicinityGraph(self.subquery, *self.__config.get_config_params())
        for node in self.__get_limited_sorted_nodes_to_visualize():
            graph_copy.add_node(node)
        return graph_copy
    
    
    def get_graph_viz(self,
            node_size: str = '2', 
            node_color: str = 'green'
            ) -> Graph:
        """
        Visualizes the graph, highlighting the vicinity terms with the highest proximity score.

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
        for index, node in enumerate(viewable_graph.get_proximity_nodes_sorted()):
            node_proximity_score = round(node.get_proximity_score(), 1)
            p_with = str(node.get_proximity_score())
            visual_graph.node("'" +str(index+1)+"'", node.get_term(), fixedsize='true', width=node_size, 
                              penwidth=p_with, color=node_color)
            visual_graph.edge('0', "'" +str(index+1)+"'", label=str(node_proximity_score), len=str(node_proximity_score))
            
        return visual_graph
    

    def get_similarity_score_as_base_graph(self, external_graph: 'VicinityGraph') -> float:
        """
        Calculate the similarity score between the self graph and another graph. To calculate it, it is 
        proposed to compare the graph using a multidimensional vector space, where the properties of 
        each term define a dimension of the space, defining the neighbour terms of the current graph 
        as the base vector.

        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be compared

        Returns
        -------
        cosine_of_angle : float
            The cosine similarity between the current graph and the external graph
        """
        # Get the dictionaries from each graph
        self_graph_dict = self.get_graph_as_dict()
        external_graph_dict = external_graph.get_graph_as_dict()
        
        # Divide the dicts into two types: proximity nodes dict and frequency nodes dict
        proximity_nodes_self_dict = {key: value for key, value in self_graph_dict.items() if value.get('criteria') == 'proximity'}
        proximity_nodes_external_dict = {key: value for key, value in external_graph_dict.items() if value.get('proximity_score') > 0}
        frequency_nodes_self_dict = {key: value for key, value in self_graph_dict.items() if value.get('criteria') == 'frequency'}
        frequency_nodes_external_dict = external_graph_dict
        
        # Validate the proximity and frequency vector lengths
        prox_nodes_length = len(proximity_nodes_self_dict.keys())
        freq_nodes_length = len(frequency_nodes_self_dict.keys())
        
        if prox_nodes_length == 0 and freq_nodes_length == 0:
            print("Warning: The base graph doesn't have proximity or frequency nodes.")
            return -1.0
        
        # Calculate the base vector with terms from the current graph
        proximity_vector_base = set(proximity_nodes_self_dict.keys()) | set(proximity_nodes_external_dict.keys())
        frequency_vector_base = set(frequency_nodes_self_dict.keys()) | set(frequency_nodes_external_dict.keys())
        
        # Initialize the vectors
        self_proximity_vector: list[float] = [] 
        external_proximity_vector: list[float] = []
        self_frequency_vector: list[float] = [] 
        external_frequency_vector: list[float] = []
        
        # Calculate the proximity vectors in the multidimensional space, i.e. generate the vector space
        for term in proximity_vector_base: 
            self_proximity_vector.append(proximity_nodes_self_dict.get(term, {}).get('proximity_score', 0))
            external_proximity_vector.append(proximity_nodes_external_dict.get(term, {}).get('proximity_score', 0))
        
        # Calculate the frequency vectors in the multidimensional space, i.e. generate the vector space
        for term in frequency_vector_base: 
            self_frequency_vector.append(frequency_nodes_self_dict.get(term, {}).get('frequency_score', 0))
            external_frequency_vector.append(frequency_nodes_external_dict.get(term, {}).get('frequency_score', 0))
        
        # Calculate the cosine of the angle between the vectors
        cosine_of_prox_angle = VectorUtils.get_cosine_between_vectors(self_proximity_vector, external_proximity_vector)
        cosine_of_freq_angle = VectorUtils.get_cosine_between_vectors(self_frequency_vector, external_frequency_vector)
        
        cosine_of_angle = cosine_of_prox_angle * prox_nodes_length + cosine_of_freq_angle * freq_nodes_length
        cosine_of_angle /= (prox_nodes_length + freq_nodes_length)
        
        return cosine_of_angle
    

    def get_union_to_graph(self,
            external_graph: 'VicinityGraph',
            sum_scores: bool = True
            ) -> 'VicinityGraph':
        """
        Unites an external graph with the own graph and obtains a new graph
        The merging process involves iterating through the nodes of both graphs, calculating 
        the sum of frequency scores and proximity scores.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be united
        sum_scores : bool, optional
            If True, then get the new score of each intersected node by adding 
            their values, else get the new score by getting the max value 
            between local and external scores.

        Returns
        -------
        united_graph : VicinityGraph
            The union between copy of the graph itself and an external graph
        """
        united_graph = self.__get_calculation_of_intersected_terms(external_graph, sum_scores)
        
        node_terms_from_copy_graph = united_graph.get_terms_str_from_all_nodes()
        node_terms_from_ext_graph = external_graph.get_terms_str_from_all_nodes()

        #if the external graph has exclusive terms, then these need to be added to the united graph
        for node_term in set(node_terms_from_ext_graph) - set(node_terms_from_copy_graph):
            node_from_ext_graph = external_graph.get_node_by_term(node_term)
            if node_from_ext_graph:    # Double check
                copy_node_from_ext_graph = copy.deepcopy(node_from_ext_graph)
                united_graph.add_node(copy_node_from_ext_graph)

        return united_graph
    

    def get_intersection_to_graph(self,
            external_graph: 'VicinityGraph',
            sum_scores: bool = True
            ) -> 'VicinityGraph':
        """
        Intersects an external graph with the own graph and obtains a new graph
        The merging process involves iterating through the nodes of both graphs, calculating 
        the sum of weights and the average distances between each one.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be intersected
        sum_scores : bool, optional
            If True, then get the new score of each intersected node by adding 
            their values, else get the new score by getting the max value 
            between local and external scores.

        Returns
        -------
        intersected_graph : VicinityGraph
            The intersection between copy of the graph itself and an external graph
        """
        #if the graph copy term is already in the external graph
        intersected_graph = self.__get_calculation_of_intersected_terms(external_graph, sum_scores)

        node_terms_from_copy_graph = intersected_graph.get_terms_str_from_all_nodes()
        node_terms_from_ext_graph = external_graph.get_terms_str_from_all_nodes()
        #if the graph copy has exclusive terms, then it needs to be deleted
        for node_term in set(node_terms_from_copy_graph) - set(node_terms_from_ext_graph):
            intersected_graph.delete_node_by_term(node_term)

        return intersected_graph
    

    def __get_calculation_of_intersected_terms(self,
            external_graph: 'VicinityGraph',
            sum_scores: bool = True
            ) -> 'VicinityGraph':
        """
        Calculates the sum or max value (depending on the sum_scores parameter) of proximity 
        and frequency scores, of the nodes between the external graph and itself.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to calculate the intersected terms
        sum_scores : bool, optional
            If True, then get the new score of each intersected node by adding 
            their values, else get the new score by getting the max value 
            between local and external scores.

        Returns
        -------
        copy_graph : VicinityGraph
            The copied graph that contain the calculation of the intersected terms
        """
        copy_graph = copy.deepcopy(self)

        for node_term in copy_graph.__get_terms_from_intersection_between_graphs(external_graph):
            node_from_copy_graph = copy_graph.get_node_by_term(node_term)
            node_from_ext_graph = external_graph.get_node_by_term(node_term)
            
            if not node_from_copy_graph or not node_from_ext_graph:   # Double check that nodes are not None values
                continue

            # initialize variables
            copy_frequency_score = node_from_copy_graph.get_frequency_score()
            copy_prox_score = node_from_copy_graph.get_proximity_score()
            ext_frequency_score = node_from_ext_graph.get_frequency_score()
            ext_prox_score = node_from_ext_graph.get_proximity_score()

            # sum (or get the max value from) the frequency scores and proximity scores
            if sum_scores:
                sum_of_prox_scores = copy_prox_score + ext_prox_score
                sum_of_frequency_scores = copy_frequency_score + ext_frequency_score
            else:
                sum_of_prox_scores = max(copy_prox_score, ext_prox_score)
                sum_of_frequency_scores = max(copy_frequency_score, ext_frequency_score)

            # round score and distance to six decimal places
            sum_of_prox_scores = round(sum_of_prox_scores, 6)
            sum_of_frequency_scores = round(sum_of_frequency_scores, 6)

            #set new frequency score and proximity score to each intersected term
            node_from_copy_graph.set_frequency_score(sum_of_frequency_scores)
            node_from_copy_graph.set_proximity_score(sum_of_prox_scores)
            
            # if the proximity score is obtained from the external node, then set the node criteria as proximity
            if copy_prox_score <= 0 and ext_prox_score > 0:
                node_from_copy_graph.set_criteria('proximity')
        
        return copy_graph
    

    def __get_terms_from_intersection_between_graphs(self,
            graph_b: 'VicinityGraph'
            ) -> set[str]:
        """
        This function returns the set of terms that exist in both the current graph and an external graph.
        It calculates the intersection of the terms from the nodes of both graphs.

        Parameters:
        graph_b (VicinityGraph): The external graph to find the intersection with the current graph.

        Returns:
        set[str]: A set containing the terms that exist in both the current graph and the external graph.
        """
        base_terms = set(self.get_terms_str_from_all_nodes()) & set(graph_b.get_terms_str_from_all_nodes())
        return base_terms
    

    def __get_limited_sorted_nodes_to_visualize(self) -> list[VicinityNode]:
        """
        Return the list of proximity nodes sorted by proximity score in descending order, and limit 
        its length by the number of graph terms, from the current graph configuration.
        It also ignores the vicinity nodes that include the query terms.

        Parameters:
        self (VicinityGraph): The instance of the VicinityGraph class.

        Returns:
        list[VicinityNode]: A list of VicinityNode objects, sorted by proximity score in descending order, 
        and limited by the number of graph terms, excluding nodes that contain the query terms.
        """
        sorted_nodes_to_visualize = self.get_proximity_nodes_sorted()
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