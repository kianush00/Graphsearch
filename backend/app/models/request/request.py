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
    

class PydanticSentenceRequest(BaseModel):
    position_in_doc: int
    raw_text: str
    neighbour_terms: List[PydanticNeighbourTermRequest]


class PydanticDocumentRequest(BaseModel):
    doc_id: str
    title: str
    abstract: str
    weight: float
    neighbour_terms: List[PydanticNeighbourTermRequest]
    sentences: List[PydanticSentenceRequest]


class PydanticRankingRequest(BaseModel):
    visible_neighbour_terms: List[PydanticNeighbourTermRequest]
    documents: List[PydanticDocumentRequest]
