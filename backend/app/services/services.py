from app.models.request.request import PydanticNeighboursTerm, PydanticNeighboursTerms
from app.models.srex.ranking import Ranking
import json


# Load Stop Words
with open('app/data/stopwords_data.json') as f:
    stopwords_data = json.load(f)
stop_words_list = stopwords_data.get('words')


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

queryService = QueryService()