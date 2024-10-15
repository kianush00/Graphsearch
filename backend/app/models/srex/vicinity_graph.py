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
    
    def __init__(self, term: str, total_ponderation: float = 0.0, proximity_ponderation: float = 0.0, 
                 distance: float = 1.0, criteria: str = "frequency"):
        """
        Initialize a new instance of the VicinityNode class.

        Parameters:
        term (str): The term of the node.
        total_ponderation (float, optional): The total ponderation of the node. Default is 0.0.
        proximity_ponderation (float, optional): The proximity ponderation of the node. Default is 0.0.
        distance (float, optional): The distance of the node. Default is 0.0.
        criteria (str, optional): The criteria of the node. Default is proximity.
        """
        self.__term = term
        self.__total_ponderation = total_ponderation
        self.__proximity_ponderation = proximity_ponderation
        self.__distance = distance
        self.set_criteria(criteria)
    

    def get_term(self) -> str:
        """
        Returns the term of the current node.

        Returns:
        str: The term of the current node.
        """
        return self.__term


    def get_total_ponderation(self) -> float:
        """
        Returns the total ponderation of the current node.

        Returns:
        float: The total ponderation of the current node.
        """
        return self.__total_ponderation


    def set_total_ponderation(self, total_ponderation: float) -> None:
        """
        Sets the total ponderation of the current node.

        Parameters:
        ponderation (float): The new total ponderation value.
        """
        self.__total_ponderation = total_ponderation
    
    
    def get_proximity_ponderation(self) -> float:
        """
        Returns the proximity ponderation of the current node.

        Returns:
        float: The proximity ponderation of the current node.
        """
        return self.__proximity_ponderation


    def set_proximity_ponderation(self, proximity_ponderation: float) -> None:
        """
        Sets the proximity ponderation of the current node.

        Parameters:
        ponderation (float): The new proximity ponderation value.
        """
        self.__proximity_ponderation = proximity_ponderation


    def get_distance(self) -> float:
        """
        Returns the distance of the current node.

        Returns:
        float: The distance of the current node.
        """
        return self.__distance


    def set_distance(self, distance: float) -> None:
        """
        Sets the distance of the current node.

        Parameters:
        distance (float): The new distance value.
        """
        self.__distance = distance
    
    
    def get_criteria(self) -> str:
        """
        Returns the criteria used to determine the type of node calculation (it can be: proximity, frequency or exclusion).
        - 'proximity': The system calculates the distance of the node to get the new ranking.
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
        if criteria == 'proximity' and self.__distance <= 0.0:
            raise ValueError("Distance cannot be negative or zero when criteria is 'proximity'.")
        
        self.__criteria = criteria


    def __str__(self) -> str:
        """
        Returns a string representation of the current node.

        Returns:
        str: A string representation of the current node.
        """
        term = self.__term
        total_ponderation = round(self.__total_ponderation, 2)
        proximity_ponderation = round(self.__proximity_ponderation, 2)
        distance = round(self.__distance, 1)
        criteria = self.__criteria
        string = f"TERM: {term} ; TOTAL_PONDERATION: {total_ponderation} ; PROXIMITY_PONDERATION: {proximity_ponderation} ; DISTANCE: {distance} ; CRITERIA: {criteria}"
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
        Return the list of all nodes (all criteria types) sorted by proximity ponderation in descending order.

        Returns:
        list[VicinityNode]: A list of VicinityNode objects sorted by proximity ponderation in descending order.
        """
        sorted_nodes = sorted(self.__nodes, key=lambda node: node.get_proximity_ponderation(), reverse=True)
        return sorted_nodes
    
    
    def get_proximity_nodes_sorted(self) -> list[VicinityNode]:
        """
        Return the list of proximity nodes sorted by proximity ponderation in descending order.

        Returns:
        list[VicinityNode]: A list of proximity VicinityNode objects sorted by proximity ponderation in descending order.
        """
        sorted_proximity_nodes = [node for node in self.__nodes if node.get_criteria() == "proximity"]
        sorted_proximity_nodes = sorted(sorted_proximity_nodes, key=lambda node: node.get_proximity_ponderation(), reverse=True)
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
        Each node is represented as a string in the format: "term ; ponderation ; distance"

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

        e.g.   {'v_term1': {'ponderation': 3.0, 'distance': 3.4}, 
                'v_term2': {'ponderation': 2.0, 'distance': 1.6}, ...}

        Returns:
        dict[str, dict[str, float]]: A dictionary representing the graph. The keys 
        are the terms, and the values are dictionaries containing the 'ponderation' 
        and 'distance' of each term.
        """
        graph_dict = {n.get_term(): {'total_ponderation': n.get_total_ponderation(),
                                     'proximity_ponderation': n.get_proximity_ponderation(),
                                     'distance': n.get_distance(),
                                     'criteria': n.get_criteria()} for n in self.get_all_nodes_sorted()}
        return graph_dict

    
    def get_viewable_graph_copy(self) -> 'VicinityGraph':
        """
        Returns a copy of the graph, with the vicinity terms with the highest ponderation 
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
        for index, node in enumerate(viewable_graph.get_proximity_nodes_sorted()):
            node_distance = round(node.get_distance(), 1)
            p_with = str(node.get_proximity_ponderation())
            visual_graph.node("'" +str(index+1)+"'", node.get_term(), fixedsize='true', width=node_size, 
                              penwidth=p_with, color=node_color)
            visual_graph.edge('0', "'" +str(index+1)+"'", label=str(node_distance), len=str(node_distance))
            
        return visual_graph
    

    def get_euclidean_distance_as_base_graph(self,
            external_graph: 'VicinityGraph',
            include_ponderation: bool = False,
            new_min_normalized_value: float = 0.5,
            new_max_normalized_value: float = 1.0
            ) -> float:
        """
        Calculate the euclidean distance with another graph. To calculate it, it is proposed to compare 
        the graph using a multidimensional vector space, where the properties of each term define a 
        dimension of the space, defining the neighbour terms of the current graph as the base vector.

        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be compared
        include_ponderation : bool, optional
            Select whether to include ponderation of the nodes as a parameter to the comparison function
        new_min_normalized_value : float, optional
            New minimum normalized value for distance and ponderation vectors
        new_max_normalized_value : float, optional
            New maximum normalized value for distance and ponderation vectors

        Returns
        -------
        distance : float
            The euclidean distance between the current graph and the external graph
        """
        def append_values_from_term(term: str, key: str, self_dict: dict, external_dict: dict, ponderation_values: bool):
            """Helper function to append values from a term based on a specific key.
            Each term is attached only if it is in both vectors.
            Also normalize the ponderation values, if the include_ponderation flag is set to true."""
            self_value: float = self_dict.get(term, {}).get(key, 0)
            external_value: float = external_dict.get(term, {}).get(key, 0)
            if self_value > 0 and external_value > 0:
                # If include_ponderation is true, then normalize the ponderation values to the new range [0.5, 1.0]
                if ponderation_values:
                    old_max = max(self_value, external_value)
                    self_value = new_min_normalized_value + (self_value * (new_max_normalized_value - new_min_normalized_value) / old_max)
                    external_value = new_min_normalized_value + (external_value * (new_max_normalized_value - new_min_normalized_value) / old_max)
                self_vector.append(self_value)
                external_vector.append(external_value)
        
        # Calculate the base vector with terms from the current graph
        vector_base = set(self.get_terms_str_from_all_nodes())
        
        # Get the dictionaries of normalized nodes from each graph 
        normalized_proximity_self_nodes = self.get_proximity_dict_with_normalized_distances(new_min_normalized_value, new_max_normalized_value)
        normalized_proximity_external_nodes = external_graph.get_proximity_dict_with_normalized_distances(new_min_normalized_value, new_max_normalized_value)
        default_frequency_self_nodes = self.get_graph_as_dict()
        default_all_external_nodes = {key: value for key, value in external_graph.get_graph_as_dict().items() if value.get('criteria') == 'frequency'}
        
        # Initialize the vectors
        self_vector: list[float] = [] 
        external_vector: list[float] = []

        # Calculate the two vectors in the multidimensional space, i.e. generate the vector space
        for term in vector_base: 
            append_values_from_term(term, 'distance', normalized_proximity_self_nodes, normalized_proximity_external_nodes, False)
        
        # Add ponderation values to the two vectors if include ponderation is True
        for term in vector_base:
            append_values_from_term(term, 'proximity_ponderation', normalized_proximity_self_nodes, normalized_proximity_external_nodes, True)
        
        for term in vector_base:
            append_values_from_term(term, 'total_ponderation', default_frequency_self_nodes, default_all_external_nodes, True)

        # Calculate the euclidean distance between the vectors if their lenghts are greater than 0, otherwise pass a big value
        if len(self_vector) > 0 and len(external_vector) > 0:
            distance = VectorUtils.get_euclidean_distance(self_vector, external_vector)
        else:
            distance = float("inf")
        
        return distance
    

    def get_proximity_dict_with_normalized_distances(self, 
            new_min_normalized_value: float = 0.5,
            new_max_normalized_value: float = 1.0
            ) -> dict[str, dict[str, float | str]]:
        """
        Return a dictionary with the normalized distances, and unchanged ponderations, of each node in the graph.
        Normalized distance values will be between the parameter values (by default between 0.5 and 1.0).
        
        Parameters
        ----------
        new_min_normalized_value : float, optional
            New minimum normalized value for distance and ponderation vectors
        new_max_normalized_value : float, optional
            New maximum normalized value for distance and ponderation vectors

        Returns
        -------
        normalized_self_nodes : dict[str, dict[str, float | str]]
            A dictionary with unchanged ponderations and normalized distances of each node in the graph
            e.g.   {'v_term1': {'ponderation': 1.4, 'distance': 0.57}, 
                    'v_term2': {'ponderation': 2.35, 'distance': 0.82}, ...}
        """
        # Get the ponderations and distances from the graph 
        normalized_self_nodes = self.get_graph_as_dict()
        normalized_self_nodes = {key: value for key, value in normalized_self_nodes.items() if value.get('criteria') == 'proximity'}

        # Get the distance and ponderation vectors from the graph
        distance_vector = [value['distance'] for value in normalized_self_nodes.values()]

        # Get the normalized version of the distance vector, in a range of [0.5, 1.0]
        normalized_distance_vector = VectorUtils.normalize_vector(distance_vector, new_min_normalized_value, 
                                            new_max_normalized_value, 1.0, self.__config.get_limit_distance())

        # Assign the normalized values ​​to the dictionary
        for index, subdict in enumerate(normalized_self_nodes.values()):
            subdict['distance'] = normalized_distance_vector[index]

        return normalized_self_nodes
    

    def get_union_to_graph(self,
            external_graph: 'VicinityGraph',
            sum_ponderations: bool = True
            ) -> 'VicinityGraph':
        """
        Unites an external graph with the own graph and obtains a new graph
        The merging process involves iterating through the nodes of both graphs, calculating 
        the sum of weights and the average distances between each one.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be united
        sum_ponderations : bool, optional
            If True, then get the new ponderation of each intersected node by adding 
            their values, else get the new ponderation by getting the max value 
            between local and external ponderations.

        Returns
        -------
        united_graph : VicinityGraph
            The union between copy of the graph itself and an external graph
        """
        united_graph = self.__get_calculation_of_intersected_terms(external_graph, sum_ponderations)
        
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
            sum_ponderations: bool = True
            ) -> 'VicinityGraph':
        """
        Intersects an external graph with the own graph and obtains a new graph
        The merging process involves iterating through the nodes of both graphs, calculating 
        the sum of weights and the average distances between each one.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to be intersected
        sum_ponderations : bool, optional
            If True, then get the new ponderation of each intersected node by adding 
            their values, else get the new ponderation by getting the max value 
            between local and external ponderations.

        Returns
        -------
        intersected_graph : VicinityGraph
            The intersection between copy of the graph itself and an external graph
        """
        #if the graph copy term is already in the external graph
        intersected_graph = self.__get_calculation_of_intersected_terms(external_graph, sum_ponderations)

        node_terms_from_copy_graph = intersected_graph.get_terms_str_from_all_nodes()
        node_terms_from_ext_graph = external_graph.get_terms_str_from_all_nodes()
        #if the graph copy has exclusive terms, then it needs to be deleted
        for node_term in set(node_terms_from_copy_graph) - set(node_terms_from_ext_graph):
            intersected_graph.delete_node_by_term(node_term)

        return intersected_graph
    

    def __get_calculation_of_intersected_terms(self,
            external_graph: 'VicinityGraph',
            sum_ponderations: bool = True
            ) -> 'VicinityGraph':
        """
        Calculates the sum of weights and the average distances of the nodes between 
        the external graph and itself.
        
        Parameters
        ----------
        external_graph : VicinityGraph
            The external graph to calculate the intersected terms
        sum_ponderations : bool, optional
            If True, then get the new ponderation of each intersected node by adding 
            their values, else get the new ponderation by getting the max value 
            between local and external ponderations.

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
            copy_distance = node_from_copy_graph.get_distance()
            copy_total_pond = node_from_copy_graph.get_total_ponderation()
            copy_prox_pond = node_from_copy_graph.get_proximity_ponderation()
            ext_distance = node_from_ext_graph.get_distance()
            ext_total_pond = node_from_ext_graph.get_total_ponderation()
            ext_prox_pond = node_from_ext_graph.get_proximity_ponderation()

            # sum (or get the max value from) the total ponderations and proximity ponderations
            if sum_ponderations:
                sum_of_prox_ponds = copy_prox_pond + ext_prox_pond
                sum_of_total_ponds = copy_total_pond + ext_total_pond
            else:
                sum_of_prox_ponds = max(copy_prox_pond, ext_prox_pond)
                sum_of_total_ponds = max(copy_total_pond, ext_total_pond)
                
            # calculate the average distance between the two nodes
            average_distance = self.__calculate_average_distance(
                copy_distance, ext_distance, copy_prox_pond, ext_prox_pond)

            # round ponderation and distance to six decimal places
            sum_of_prox_ponds = round(sum_of_prox_ponds, 6)
            sum_of_total_ponds = round(sum_of_total_ponds, 6)
            average_distance = round(average_distance, 6)

            #set new distance and ponderation to each intersected term
            node_from_copy_graph.set_distance(average_distance)
            node_from_copy_graph.set_total_ponderation(sum_of_total_ponds)
            node_from_copy_graph.set_proximity_ponderation(sum_of_prox_ponds)
            
            # if the distance is obtained from the external node, then set the node criteria as proximity
            if copy_distance <= 0 and ext_distance > 0:
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
    
    
    def __calculate_average_distance(self, 
        copy_distance: float, 
        ext_distance: float, 
        copy_prox_pond: float, 
        ext_prox_pond: float
        ) -> float:
        """
        Calculates the average distance from the intersection between two nodes.
        
        Parameters:
        copy_distance (float): The distance of the node from the current graph.
        ext_distance (float): The distance of the node from the external graph.
        copy_prox_pond (float): The proximity ponderation of the node from the current graph.
        ext_prox_pond (float): The proximity ponderation of the node from the external graph.
        node_from_copy_graph (VicinityNode): The node from the current graph.
        
        Returns:
        float: The average distance of the intersection between the two nodes. If both nodes are of frequency type, 
        it returns -1.0. If the copy node is of proximity type and the external node is of frequency type, it returns 
        the copy node's distance. If the copy node is of frequency type and the external node is of proximity type, 
        it returns the external node's distance and sets the criteria of the copy node to 'proximity'. If both nodes 
        are of proximity type, it calculates the average distance using the proximity ponderations.
        """
        # calculate the average distance
        if copy_distance <= 0 or ext_distance <= 0:
            if copy_distance <= 0 and ext_distance <= 0:   # if both nodes are frequency type
                average_distance = -1.0
            elif copy_distance > 0:  # if the copy node is proximity type and the external node is frequency type
                average_distance = copy_distance
            else:  # if the copy node is frequency type and the external node is proximity type
                average_distance = ext_distance
        else:   # if both nodes are proximity type
            average_distance = ((copy_distance * copy_prox_pond) + (ext_distance * ext_prox_pond)) / (copy_prox_pond + ext_prox_pond)

        return average_distance

    

    def __get_limited_sorted_nodes_to_visualize(self) -> list[VicinityNode]:
        """
        Return the list of proximity nodes sorted by ponderation in descending order, and limit 
        its length by the number of graph terms, from the current graph configuration.
        It also ignores the vicinity nodes that include the query terms.

        Parameters:
        self (VicinityGraph): The instance of the VicinityGraph class.

        Returns:
        list[VicinityNode]: A list of VicinityNode objects, sorted by ponderation in descending order, 
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