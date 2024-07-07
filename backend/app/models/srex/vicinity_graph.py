import re
import copy
from graphviz import Graph
from utils.vector_utils import VectorUtils




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
            self_vector.append(normalized_self_nodes.get(term, {}).get('distance', 0))
            external_vector.append(normalized_external_nodes.get(term, {}).get('distance', 0))
        
        # Add ponderation values to the two vectors if include ponderation is True
        if include_ponderation:
            for term in vector_base: 
                self_vector.append(normalized_self_nodes.get(term, {}).get('ponderation', 0))
                external_vector.append(normalized_external_nodes.get(term, {}).get('ponderation', 0))

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

            # round ponderation and distance to six decimal places
            sum_of_ponds = round(sum_of_ponds, 6)
            average_distance = round(average_distance, 6)

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