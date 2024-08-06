from pydantic import BaseModel
from typing import List


class QueryTerm(BaseModel):
    query: str
    search_results: int
    limit_distance: int
    graph_terms: int


class PydanticNeighbourTermRequest(BaseModel):
    term: str
    ponderation: float
    distance: float
    

class PydanticDocumentRequest(BaseModel):
    doc_id: str
    title: str
    abstract: str
    initial_ranking_position: int
    neighbour_terms: List[PydanticNeighbourTermRequest]


class PydanticRankingRequest(BaseModel):
    visible_neighbour_terms: List[PydanticNeighbourTermRequest]
    documents: List[PydanticDocumentRequest]
