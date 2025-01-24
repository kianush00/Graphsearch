from pydantic import BaseModel
from typing import List


class QueryTerm(BaseModel):
    query: str
    search_results: int
    limit_distance: int
    graph_terms: int
    selected_categories: List[str]


class PydanticNeighbourTermRequest(BaseModel):
    term: str
    proximity_score: float
    frequency_score: float
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
    visible_neighbour_terms: List[PydanticNeighbourTermRequest]
    documents: List[PydanticDocumentRequest]
