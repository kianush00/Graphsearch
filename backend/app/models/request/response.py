from pydantic import BaseModel
from typing import List


class RerankNewPositions(BaseModel):
    ranking_new_positions: list[int]


class PydanticNeighbourTerm(BaseModel):
    term: str
    proximity_ponderation: float
    total_ponderation: float
    distance: float
    criteria: str

    
class PydanticSentence(BaseModel):
    position_in_doc: int
    raw_text: str
    all_neighbour_terms: List[PydanticNeighbourTerm]
    

class PydanticDocument(BaseModel):
    doc_id: str
    title: str
    abstract: str
    preprocessed_text: str
    weight: float
    all_neighbour_terms: List[PydanticNeighbourTerm]
    sentences: List[PydanticSentence]


class PydanticRanking(BaseModel):
    visible_neighbour_terms: List[PydanticNeighbourTerm]
    complete_neighbour_terms: List[PydanticNeighbourTerm]
    documents: List[PydanticDocument]