from models.request.request import PydanticNeighboursTerm, PydanticNeighboursTerms
from models.srex.ranking import Ranking
from utils.data_utils import DataUtils
from fastapi import HTTPException



stop_words_list = DataUtils.load_stopwords()


class QueryService:
    def __init__(self):
        pass

    def processQuery(self, query: str) -> PydanticNeighboursTerms:
        nr_search_results        = 10
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        nr_of_graph_terms        = 8
        limit_distance           = 4 
        include_query_terms  = True
        
        try:
            ranking = Ranking(query, nr_search_results, ranking_weight_type, stop_words_list, lema, stem)
            ranking.calculate_ieee_xplore_ranking()
            ranking.generate_all_graphs(nr_of_graph_terms, limit_distance, include_query_terms, summarize)
            
            neighbour_terms = [
                PydanticNeighboursTerm(
                    term=k, 
                    frequency=v.get('ponderation'), 
                    distance=v.get('distance')
                    ) for k, v in ranking.get_graph().get_viewable_graph_copy().get_graph_as_dict().items()
            ]

            return PydanticNeighboursTerms(neighbour_terms=neighbour_terms)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Results not found: {e}")


queryService = QueryService()



