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
        ranking_weight_type  = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                 = True
        stem                 = False
        summarize            = 'mean'   # it can be: 'mean' or 'median'
        include_query_terms  = False
        
        try:
            # Generate ranking graphs from the results of the search
            ranking = Ranking(query_text, nr_search_results, ranking_weight_type, stop_words_list, lema, stem)
            ranking.calculate_ieee_xplore_ranking()
            ranking.generate_all_graphs(nr_of_graph_terms, limit_distance, include_query_terms, summarize)
            
            # Get visible neighbour terms and complete neighbour terms as PydanticNeighbourTerm objects.
            visible_neighbour_terms = self.__get_pydantic_neighbour_term_list(ranking.get_graph().get_viewable_graph_copy().get_graph_as_dict())
            complete_neighbour_terms = self.__get_pydantic_neighbour_term_list(ranking.get_graph().get_graph_as_dict())
            
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
        The ranking positions are determined based on the similarity scores.

        Parameters:
        - ranking (PydanticRankingRequest): The input ranking object containing visible neighbour terms and documents.

        Returns:
        - RerankNewPositions: An object containing the new ranking positions for each document.

        Raises:
        - HTTPException: If there is a bad request (status code 400).
        """
        try:
            # Initialize parameters
            include_ponderation = False
            
            # Initialize the visible graph
            visible_graph = self.__get_graph_from_pydantic_neighbour_term_list(ranking.visible_neighbour_terms)
            
            # Initialize a list of sentence graphs from the documents of the ranking
            document_rankpos_graph_tuple_list = self.__get_rankpos_and_sentence_graphs_from_documents(ranking)
            
            # Create similarity list
            similarity_ranking: list[float] = self.__calculate_similarity_ranking(document_rankpos_graph_tuple_list, visible_graph, include_ponderation)
            
            # Get sorted similarity list
            rank_new_positions = VectorUtils.get_sorted_positions(similarity_ranking)
            return RerankNewPositions(ranking_new_positions=rank_new_positions)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad request: {e}")
    
    
    def __get_rankpos_and_sentence_graphs_from_documents(self, ranking: PydanticRankingRequest) -> list[tuple[float, list[VicinityGraph]]]:
        """
        This function extracts the document weight and sentence neighbour term graphs from a PydanticRankingRequest object.

        Parameters:
        - ranking (PydanticRankingRequest): A PydanticRankingRequest object containing a list of documents. Each document has a weight, 
        a title, an abstract, a list of sentences, and each sentence has a position in the document and a list of neighbour terms.
    
        Returns:
        - list[tuple[float, list[VicinityGraph]]]: A list of tuples. Each tuple contains a document weight (float) and a list of 
        VicinityGraph objects representing the neighbour terms of the document's sentences.
        """
        document_rankpos_graph_tuple_list = []
        for document in ranking.documents:
            # Get the document weight and its sentence graphs list
            document_sentence_graph_list = [
                self.__get_graph_from_pydantic_neighbour_term_list(s.neighbour_terms) for s in document.sentences
            ]
            document_rankpos_graph_tuple_list.append( (document.weight, document_sentence_graph_list) )
        
        return document_rankpos_graph_tuple_list
    
    
    def __calculate_similarity_ranking(self, 
        document_rankpos_graph_tuple_list: list[tuple[float, list[VicinityGraph]]], 
        visible_graph: VicinityGraph, 
        include_ponderation: bool
        ) -> list[float]:
        """
        Calculates the similarity ranking between the visible neighbour terms and the neighbour terms of each document.
        
        Parameters:
        - document_rankpos_graph_tuple_list (list[tuple[float, list[VicinityGraph]]]): A list of tuples, where each tuple contains 
        a document weight (float) and a list of VicinityGraph objects representing the neighbour terms of the document's sentences.
        - visible_graph (VicinityGraph): A VicinityGraph object representing the neighbour terms of the visible graph.
        - include_ponderation (bool): A flag indicating whether to include ponderation in the cosine similarity calculation.
    
        Returns:
        - list[float]: A list of similarity scores representing the ranking of the documents based on their similarity 
        to the visible neighbour terms.
        """
        # Initialize the similarity scores
        similarity_ranking: list[float] = []
        
        for doc_weight, doc_sentence_graph_list in document_rankpos_graph_tuple_list:
            # Calculate the cosine similarity between the visible graph and each sentence graph in the document
            cosine_similarity_by_graph: list[float] = [
                (doc_weight * 0.05) + (doc_weight * visible_graph.get_cosine_similarity(graph, include_ponderation)) for graph in doc_sentence_graph_list
            ]
            
            # Sort the list of similarity scores and get the first two scores
            cosine_similarity_by_graph_top_2 = sorted(cosine_similarity_by_graph, reverse=True)[:2]
            
            # Get the average between the two scores
            if len(cosine_similarity_by_graph_top_2) < 1:
                cosine_similarity_avg_top_2 = 0
            if len(cosine_similarity_by_graph_top_2) < 2:
                cosine_similarity_avg_top_2 = cosine_similarity_by_graph_top_2[0]
            else:
                cosine_similarity_avg_top_2 = (cosine_similarity_by_graph_top_2[0] + cosine_similarity_by_graph_top_2[1]) / 2
                
            # Add the average similarity score to the ranking list
            similarity_ranking.append(cosine_similarity_avg_top_2)
            
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
            _weight = d.get_weight()
            
            # Get the document sentences and their neighbour terms
            _sentences: list[PydanticSentence] = []
            for s in d.get_sentences():
                _position_in_doc = s.get_position_in_doc()
                _raw_text = s.get_raw_text()
                sentence_neighbour_terms = self.__get_pydantic_neighbour_term_list(s.get_graph().get_graph_as_dict())
                _sentences.append(PydanticSentence(position_in_doc=_position_in_doc, raw_text=_raw_text, 
                                                    neighbour_terms=sentence_neighbour_terms))
            
            # Add the document to the list of documents with their neighbour terms and sentences
            documents.append(PydanticDocument(doc_id=_doc_id, title=_title, abstract=_abstract, 
                    weight=_weight, neighbour_terms=doc_neighbour_terms, sentences=_sentences))
        
        return documents
    
    
    
    def __get_pydantic_neighbour_term_list(self, graph_dict: dict[str, dict[str, float]]) -> list[PydanticNeighbourTerm]:
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
            ponderation=v.get('ponderation'), 
            distance=v.get('distance')
            ) for k, v in graph_dict.items()
        ]
    
    
    def __get_graph_from_pydantic_neighbour_term_list(self, neighbour_terms: List[PydanticNeighbourTermRequest]) -> VicinityGraph:
        """
        This function converts a list of PydanticNeighbourTermRequest objects into a VicinityGraph object.

        Parameters:
        - neighbour_terms (List[PydanticNeighbourTermRequest]): A list of PydanticNeighbourTermRequest objects, where each 
        object represents a neighbour term with its attributes.

        Returns:
        - VicinityGraph: A VicinityGraph object, where each neighbour term from the input list is added as a node to the graph.
        """
        graph = VicinityGraph(subquery="new")
        for node in neighbour_terms:
            graph.add_node(VicinityNode(term=node.term, ponderation=node.ponderation, distance=node.distance))
        return graph



queryService = QueryService()

