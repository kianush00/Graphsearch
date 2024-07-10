from models.request.response import PydanticRanking, PydanticDocument, PydanticNeighbourTerm
from models.srex.ranking import Ranking
from utils.data_utils import DataUtils
from fastapi import HTTPException



stop_words_list = DataUtils.load_stopwords()


class QueryService:
    def __init__(self):
        pass

    def processQuery(self, query: str) -> PydanticRanking:
        nr_search_results        = 10
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        nr_of_graph_terms        = 10
        limit_distance           = 4 
        include_query_terms  = True
        
        try:
            ranking = Ranking(query, nr_search_results, ranking_weight_type, stop_words_list, lema, stem)
            ranking.calculate_ieee_xplore_ranking()
            ranking.generate_all_graphs(nr_of_graph_terms, limit_distance, include_query_terms, summarize)
            
            # Get visible neighbour terms and complete neighbour terms as PydanticNeighbourTerm objects.
            visible_neighbour_terms = [
                PydanticNeighbourTerm(
                    term=k, 
                    ponderation=v.get('ponderation'), 
                    distance=v.get('distance')
                    ) for k, v in ranking.get_graph().get_viewable_graph_copy().get_graph_as_dict().items()
            ]
            
            complete_neighbour_terms = [
                PydanticNeighbourTerm(
                    term=k, 
                    ponderation=v.get('ponderation'), 
                    distance=v.get('distance')
                    ) for k, v in ranking.get_graph().get_graph_as_dict().items()
            ]
            
            # Get documents and its neighbourterms
            documents = []
            for d in ranking.get_documents():
                doc_neighbour_terms = [
                    PydanticNeighbourTerm(
                    term=k, 
                    ponderation=v.get('ponderation'), 
                    distance=v.get('distance')
                    ) for k, v in d.get_graph().get_viewable_graph_copy().get_graph_as_dict().items()
                ]
                _doc_id = d.get_doc_id()
                _title = d.get_title()
                _abstract = d.get_abstract()
                documents.append(PydanticDocument(doc_id=_doc_id, title=_title, abstract=_abstract, neighbour_terms=doc_neighbour_terms))
                

            return PydanticRanking(visible_neighbour_terms=visible_neighbour_terms, 
                                   complete_neighbour_terms=complete_neighbour_terms,
                                   documents=documents)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Results not found: {e}")


queryService = QueryService()



