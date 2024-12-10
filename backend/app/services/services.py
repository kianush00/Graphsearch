from models.request.request import PydanticRankingRequest, PydanticNeighbourTermRequest
from models.request.response import PydanticRanking, PydanticDocument, PydanticSentence, PydanticNeighbourTerm, RerankNewPositions
from models.srex.ranking import Ranking
from models.srex.vicinity_graph import VicinityGraph, VicinityNode
from utils.data_utils import DataUtils
from utils.vector_utils import VectorUtils
from fastapi import HTTPException
from typing import List



stop_words_list = DataUtils.load_stopwords()


class QueryService:
    def process_query(self, query_text: str, nr_search_results: int, limit_distance: int, nr_of_graph_terms: int) -> PydanticRanking:
        """
        Processes a query and generates a PydanticRanking object containing visible neighbour terms, complete neighbour terms,
        and documents with their neighbour terms.

        Parameters:
        - query_text (str): The input query text.
        - nr_search_results (int): The number of search results to retrieve.
        - limit_distance (int): The maximum distance for neighbour terms.
        - nr_of_graph_terms (int): The number of neighbour terms to include in the user graph.

        Returns:
        - PydanticRanking: An object containing the processed ranking information.

        Raises:
        - HTTPException: If the results are not found (status code 404).
        """
        # Initialize parameters
        ranking_weight_type : str = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                : bool  = True
        stem                : bool  = False
        summarize           : str  = 'mean'   # it can be: 'mean' or 'median'
        include_query_terms : bool  = False
        
        try:
            # Generate ranking graphs from the results of the search
            ranking = Ranking(query_text, nr_search_results, ranking_weight_type, stop_words_list, lema, stem)
            ranking.calculate_ieee_xplore_ranking()
            ranking.generate_all_graphs(nr_of_graph_terms, limit_distance, include_query_terms, summarize)
            
            # Return the Pydantic ranking
            return self.__get_pydantic_ranking(ranking)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Results not found: {e}")
    
    
    def process_query_example_1(self) -> PydanticRanking:
        """
        Processes a query and generates a PydanticRanking object containing visible neighbour terms, complete neighbour terms,
        and documents with their neighbour terms.

        Parameters:
        - query_text (str): The input query text.
        - nr_search_results (int): The number of search results to retrieve.
        - limit_distance (int): The maximum distance for neighbour terms.
        - nr_of_graph_terms (int): The number of neighbour terms to include in the user graph.

        Returns:
        - PydanticRanking: An object containing the processed ranking information.

        Raises:
        - HTTPException: If the results are not found (status code 404).
        """
        # Initialize parameters
        query_text               = 'iot'
        nr_search_results        = 3
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        nr_of_graph_terms        = 5
        limit_distance           = 4 
        include_query_terms      = False
        
        try:
            # Generate ranking graphs from the results of the search
            document_1 = 'Securing IoT Devices and Connecting the Dots Using REST API and Middleware'
            document_2 = 'Therefore it is required to introduce a secure IoT system which doesnt allow attackers infiltration in the network through IoT devices and also to secure data in transit from IoT devices to cloud.'
            document_3 = 'Internet of Things (IoT) is a fairly disruptive technology with inconceivable growth, impact, and capability.'

            articles_list = [
                {'title': document_1, 'article_number': '1'}, 
                {'title': document_2, 'article_number': '2'}, 
                {'title': document_3, 'article_number': '3'}
            ]

            ranking = Ranking(query_text, nr_search_results, ranking_weight_type, stop_words_list, lema, stem)
            ranking.calculate_article_dictionaries_list(articles_list)
            ranking.generate_all_graphs(nr_of_graph_terms, limit_distance, include_query_terms, summarize)
            
            # Return the Pydantic ranking
            return self.__get_pydantic_ranking(ranking)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Results not found: {e}")
    
    
    def process_query_example_2(self) -> PydanticRanking:
        """
        Processes a query and generates a PydanticRanking object containing visible neighbour terms, complete neighbour terms,
        and documents with their neighbour terms.

        Parameters:
        - query_text (str): The input query text.
        - nr_search_results (int): The number of search results to retrieve.
        - limit_distance (int): The maximum distance for neighbour terms.
        - nr_of_graph_terms (int): The number of neighbour terms to include in the user graph.

        Returns:
        - PydanticRanking: An object containing the processed ranking information.

        Raises:
        - HTTPException: If the results are not found (status code 404).
        """
        # Initialize parameters
        query_text               = 'iot OR sensor'
        nr_search_results        = 3
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        nr_of_graph_terms        = 5
        limit_distance           = 4
        include_query_terms      = False
        
        try:
            # Generate ranking graphs from the results of the search
            document_1 = 'In scientific research, sensors are considered as a prospective field for innovation.'
            document_2 = 'IoT sensors are efficiently used in various IoT applications for creating a smart environment by collecting real time data.'
            document_3 = 'Internet of Things (IoT) is revolutionizing our world with trillions of sensors and actuators by creating a smart environment around us.'

            articles_list = [
                {'title': document_1, 'article_number': '1'}, 
                {'title': document_2, 'article_number': '2'}, 
                {'title': document_3, 'article_number': '3'}
            ]

            ranking = Ranking(query_text, nr_search_results, ranking_weight_type, stop_words_list, lema, stem)
            ranking.calculate_article_dictionaries_list(articles_list)
            ranking.generate_all_graphs(nr_of_graph_terms, limit_distance, include_query_terms, summarize)
            
            # Return the Pydantic ranking
            return self.__get_pydantic_ranking(ranking)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Results not found: {e}")
    
    
    def get_new_ranking_positions(self, ranking: PydanticRankingRequest) -> RerankNewPositions:
        """
        This function calculates the new ranking positions for a given PydanticRankingRequest object.
        It creates a similarity ranking between the visible neighbour terms of the input ranking and the neighbour terms of each document.
        The ranking positions are determined based on the similarity scores (shortest euclidean distances between graphs).

        Parameters:
        - ranking (PydanticRankingRequest): The input ranking object containing visible neighbour terms and documents.

        Returns:
        - RerankNewPositions: An object containing the new ranking positions for each document.

        Raises:
        - HTTPException: If there is a bad request (status code 400).
        """
        try:
            # Initialize the visible graph
            visible_graph = self.__get_graph_from_pydantic_visible_neighbour_term_list(ranking.visible_neighbour_terms)
            
            # Initialize a tuple list of graphs, weights and preprocessed text from each document of the ranking
            document_weight_graph_tuple_list = self.__get_doc_weight_and_sentence_graphs_from_documents(ranking)
            
            # Create similarity scores list
            similarity_scores: list[float] = self.__calculate_similarity_scores(document_weight_graph_tuple_list, visible_graph)
            
            # Get sorted similarity list in descending order (top proximity and frequency scores)
            rank_new_positions = VectorUtils.get_positions_sorted_desc(similarity_scores)
            return RerankNewPositions(ranking_new_positions=rank_new_positions)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad request: {e}")
    
    
    def __get_doc_weight_and_sentence_graphs_from_documents(self, ranking: PydanticRankingRequest) -> list[tuple[float, VicinityGraph]]:
        """
        This function extracts the document weight and all its neighbour terms from a PydanticRankingRequest object.

        Parameters:
        - ranking (PydanticRankingRequest): A PydanticRankingRequest object containing a list of documents. Each document
        has a weight and neighbour terms.
    
        Returns:
        - list[tuple[float, VicinityGraph]]: A list of tuples. Each tuple contains a document weight (float), and a
        VicinityGraph object representing all the neighbour terms of the document.
        """
        document_weight_graph_tuple_list = []
        for document in ranking.documents:
            # Get the document weight and all its neighbour terms
            document_graph = self.__get_graph_from_pydantic_neighbour_term_list(document.all_neighbour_terms)
            document_weight_graph_tuple_list.append( (document.weight, document_graph) )
        
        return document_weight_graph_tuple_list
    
    
    def __calculate_similarity_scores(self, 
        document_weight_graph_tuple_list: list[tuple[float, VicinityGraph]], 
        visible_graph: VicinityGraph
        ) -> list[float]:
        """
        Calculates the similarity scores between the visible neighbour terms and the neighbour terms of each document, until 
        obtaining a score similarity list for each document. Similarity scores correspond to the top proximity and 
        frequency scores associated with the user graph terms.
        
        Parameters:
        - document_weight_graph_tuple_list (list[tuple[float, VicinityGraph]]): A list of tuples, where each tuple contains 
        a document weight (float) and a VicinityGraph object representing the neighbour terms of the document.
        - visible_graph (VicinityGraph): A VicinityGraph object representing the neighbour terms of the visible graph.
    
        Returns:
        - list[float]: A list of similarity scores representing the ranking of the documents based on their similarity 
        to the visible neighbour terms.
        """
        # Initialize the similarity scores
        similarity_ranking: list[float] = []

        for doc_weight, doc_graph in document_weight_graph_tuple_list:
            # Calculate the similarity score between the visible graph and the document graph
            similarity_score_between_graphs = (doc_weight * 0.0000000001) + visible_graph.get_similarity_score_as_base_graph(doc_graph)
            # Add the similarity score to the ranking list
            similarity_ranking.append(similarity_score_between_graphs)
            
        return similarity_ranking
    
    
    def __get_graph_from_pydantic_neighbour_term_list(self, neighbour_terms: List[PydanticNeighbourTermRequest]) -> VicinityGraph:
        """
        This function converts a list of PydanticNeighbourTermRequest objects into a VicinityGraph object.
        Each node contains the following attributes: term, ponderation and distance.

        Parameters:
        - neighbour_terms (List[PydanticNeighbourTermRequest]): A list of PydanticNeighbourTermRequest objects, where 
        each object represents a neighbour term with its attributes.

        Returns:
        - VicinityGraph: A VicinityGraph object, where each neighbour term from the input list is added as a node to the graph.
        """
        graph = VicinityGraph(subquery="new")
        for node in neighbour_terms:
            graph.add_node(VicinityNode(term=node.term, proximity_score=node.proximity_score, 
                                        frequency_score=node.frequency_score, criteria=node.criteria))
        return graph
    
    
    def __get_graph_from_pydantic_visible_neighbour_term_list(self, neighbour_terms: List[PydanticNeighbourTermRequest]) -> VicinityGraph:
        """
        This function converts a list of PydanticNeighbourTermRequest objects into a VicinityGraph object.
        Each node contains the following attributes: term, ponderation and distance.

        Parameters:
        - neighbour_terms (List[PydanticNeighbourTermRequest]): A list of PydanticNeighbourTermRequest objects, where 
        each object represents a neighbour term with its attributes.

        Returns:
        - VicinityGraph: A VicinityGraph object, where each neighbour term from the input list is added as a node to the graph.
        """
        graph = VicinityGraph(subquery="new")
        for node in neighbour_terms:
            graph.add_node(VicinityNode(term=node.term, proximity_score=1.0, frequency_score=1.0, criteria=node.criteria))
        return graph
    
    
    def __get_pydantic_documents_from_ranking(self, ranking: Ranking) -> list[PydanticDocument]:
        """
        Converts a Ranking object into a list of PydanticDocument objects. Each PydanticDocument object represents a document
        from the ranking, containing its neighbour terms, sentences, and other attributes.

        Parameters:
        - ranking (Ranking): An instance of the Ranking class, which contains a list of documents.

        Returns:
        - list[PydanticDocument]: A list of PydanticDocument objects, where each object represents a document from the ranking.
       """
        documents: list[PydanticDocument] = []
        
        for d in ranking.get_documents():
            # Get each document and its neighbour terms
            doc_neighbour_terms = self.__get_pydantic_neighbour_term_list(d.get_graph().get_graph_as_dict())
            _doc_id = d.get_doc_id()
            _title = d.get_title()
            _abstract = d.get_abstract()
            _preprocessed_text = d.get_preprocessed_text()
            _weight = d.get_weight()
            
            # Get the document sentences and their neighbour terms
            _sentences: list[PydanticSentence] = []
            for s in d.get_sentences():
                _position_in_doc = s.get_position_in_doc()
                _raw_text = s.get_raw_text()
                sentence_neighbour_terms = self.__get_pydantic_neighbour_term_list(s.get_graph().get_graph_as_dict())
                _sentences.append(PydanticSentence(position_in_doc=_position_in_doc, raw_text=_raw_text, 
                                                    all_neighbour_terms=sentence_neighbour_terms))
            
            # Add the document to the list of documents with their neighbour terms and sentences
            documents.append(PydanticDocument(doc_id=_doc_id, title=_title, abstract=_abstract, preprocessed_text=_preprocessed_text,
                    weight=_weight, all_neighbour_terms=doc_neighbour_terms, sentences=_sentences))
        
        return documents
    
    
    def __get_pydantic_neighbour_term_list(self, graph_dict: dict[str, dict[str, float | str]]) -> list[PydanticNeighbourTerm]:
        """
        Converts a dictionary of neighbour terms and their attributes into a list of PydanticNeighbourTerm objects.

        Parameters:
        - graph_dict (dict[str, dict[str, float | str]]): A dictionary where the keys are neighbour terms and the values are 
        dictionaries containing the term's attributes. Each attribute dictionary contains 'proximity_score',
        'frequency_score' and 'criteria' keys.

        Returns:
        - list[PydanticNeighbourTerm]: A list of PydanticNeighbourTerm objects, where each object represents a 
        neighbour term with its attributes.
        """
        return [
            PydanticNeighbourTerm(
            term=k, 
            proximity_score=v.get('proximity_score'),
            frequency_score=v.get('frequency_score'),
            criteria=v.get('criteria')
            ) for k, v in graph_dict.items()
        ]
    
    
    def __get_pydantic_ranking(self, ranking: Ranking) -> PydanticRanking:
        """
        This function converts a Ranking object into a PydanticRanking object.
        Each PydanticRanking object represents a ranking with its visible neighbour terms,
        complete neighbour terms, and documents.
        
        Parameters:
        - ranking (Ranking): An instance of the Ranking class, which contains a list of documents.
        
        Returns:
        - PydanticRanking: A PydanticRanking object, where each object represents a ranking.
        The object contains visible neighbour terms, complete neighbour terms, and documents.
        """
        # Get visible neighbour terms and complete neighbour terms as PydanticNeighbourTerm objects.
        visible_graph_dict = ranking.get_graph().get_viewable_graph_copy().get_graph_as_dict()
        complete_graph_dict = ranking.get_graph().get_graph_as_dict()

        visible_neighbour_terms = self.__get_pydantic_neighbour_term_list(visible_graph_dict)
        complete_neighbour_terms = self.__get_pydantic_neighbour_term_list(complete_graph_dict)

        # Get documents and its neighbour terms
        documents: list[PydanticDocument] = self.__get_pydantic_documents_from_ranking(ranking)

        return PydanticRanking(visible_neighbour_terms=visible_neighbour_terms, 
                                complete_neighbour_terms=complete_neighbour_terms,
                                documents=documents)





queryService = QueryService()

