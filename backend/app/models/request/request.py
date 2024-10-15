from pydantic import BaseModel
from typing import List


class QueryTerm(BaseModel):
    query: str
    search_results: int
    limit_distance: int
    graph_terms: int


class PydanticVisibleNeighbourTermRequest(BaseModel):
    term: str
    proximity_ponderation: float
    total_ponderation: float
    criteria: str


class PydanticNeighbourTermRequest(BaseModel):
    term: str
    proximity_ponderation: float
    total_ponderation: float
    distance: float
    criteria: str
    

class PydanticSentenceRequest(BaseModel):
    position_in_doc: int
    raw_text: str
    all_neighbour_terms: List[PydanticNeighbourTermRequest]


class PydanticDocumentRequest(BaseModel):
    doc_id: str
    title: str
    abstract: str
    preprocessed_text: str
    weight: float
    all_neighbour_terms: List[PydanticNeighbourTermRequest]


class PydanticRankingRequest(BaseModel):
    visible_neighbour_terms: List[PydanticVisibleNeighbourTermRequest]
    documents: List[PydanticDocumentRequest]
