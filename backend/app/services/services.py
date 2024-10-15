from models.request.request import PydanticRankingRequest, PydanticVisibleNeighbourTermRequest, PydanticNeighbourTermRequest
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
            # Initialize parameters
            include_ponderation = True
            
            # Initialize the visible graph
            visible_graph = self.__get_visible_graph_from_pydantic_neighbour_term_list(ranking.visible_neighbour_terms)
            
            # Initialize a tuple list of graphs, weights and preprocessed text from each document of the ranking
            document_weight_graph_tuple_list = self.__get_doc_weight_and_sentence_graphs_from_documents(ranking)
            
            # Create similarity scores list
            similarity_scores: list[float] = self.__calculate_similarity_scores(document_weight_graph_tuple_list, visible_graph, include_ponderation)
            
            # Get sorted similarity list in ascending order (shortest distance between vectors of the graphs)
            rank_new_positions = VectorUtils.get_positions_sorted_asc(similarity_scores)
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
        - list[tuple[float, VicinityGraph]]: A list of tuples. Each tuple contains a document weight (float), a
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
        visible_graph: VicinityGraph, 
        include_ponderation: bool
        ) -> list[float]:
        """
        Calculates the similarity scores between the visible neighbour terms and the neighbour terms of each document, until 
        obtaining the documents with the closest neighbour terms to the query terms. Similarity scores correspond to the 
        Euclidean distance between two graphs, being more similar when the distance between is smaller and vice versa.
        
        Parameters:
        - document_weight_graph_tuple_list (list[tuple[float, VicinityGraph]]): A list of tuples, where each tuple contains 
        a document weight (float) and a VicinityGraph object representing the neighbour terms of the document.
        - visible_graph (VicinityGraph): A VicinityGraph object representing the neighbour terms of the visible graph.
        - include_ponderation (bool): A flag indicating whether to include ponderation in the cosine similarity calculation.
    
        Returns:
        - list[float]: A list of similarity scores representing the ranking of the documents based on their similarity 
        to the visible neighbour terms.
        """
        # Initialize the similarity scores
        similarity_ranking: list[float] = []

        for doc_weight, doc_graph in document_weight_graph_tuple_list:
            # Calculate the euclidean distance between the visible graph and the document graph
            euclidean_distance_between_graphs = - (doc_weight * 0.01) + visible_graph.get_euclidean_distance_as_base_graph(doc_graph, include_ponderation)
                
            # Add the first similarity score to the ranking list
            similarity_ranking.append(euclidean_distance_between_graphs)
            
        return similarity_ranking
    
    
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
        - graph_dict (dict[str, dict[str, float]]): A dictionary where the keys are neighbour terms and the values are 
        dictionaries containing the term's attributes. Each attribute dictionary contains 'ponderation' and 'distance' keys.

        Returns:
        - list[PydanticNeighbourTerm]: A list of PydanticNeighbourTerm objects, where each object represents a 
        neighbour term with its attributes.
        """
        return [
            PydanticNeighbourTerm(
            term=k, 
            proximity_ponderation=v.get('proximity_ponderation'),
            total_ponderation=v.get('total_ponderation'),
            distance=v.get('distance'),
            criteria=v.get('criteria')
            ) for k, v in graph_dict.items()
        ]
    
    
    def __get_visible_graph_from_pydantic_neighbour_term_list(self, neighbour_terms: List[PydanticVisibleNeighbourTermRequest]) -> VicinityGraph:
        """
        This function converts a list of PydanticVisibleNeighbourTermRequest objects into a VicinityGraph object.
        A distance of 1 is added to the nodes since the nodes with the "proximity" criterion will be compared 
        with the ranking documents.
        Each node contains the following attributes: term, ponderation, distance of 1.0 and criteria.

        Parameters:
        - neighbour_terms (List[PydanticVisibleNeighbourTermRequest]): A list of PydanticVisibleNeighbourTermRequest 
        objects, where each object represents a neighbour term with its attributes.

        Returns:
        - VicinityGraph: A VicinityGraph object, where each neighbour term from the input list is added as a node to the graph.
        """
        graph = VicinityGraph(subquery="new")
        for node in neighbour_terms:
            graph.add_node(VicinityNode(
                term=node.term, proximity_ponderation=node.proximity_ponderation, total_ponderation=node.total_ponderation, 
                distance=1.0, criteria=node.criteria))
        return graph
    
    
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
            graph.add_node(VicinityNode(term=node.term, proximity_ponderation=node.proximity_ponderation, 
                    total_ponderation=node.total_ponderation, distance=node.distance, criteria=node.criteria))
        return graph



queryService = QueryService()

