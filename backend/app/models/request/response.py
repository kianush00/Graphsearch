from pydantic import BaseModel
from typing import List


class RerankNewPositions(BaseModel):
    ranking_new_positions: list[int]


class PydanticNeighbourTerm(BaseModel):
    term: str
    ponderation: float
    distance: float

    
class PydanticSentence(BaseModel):
    position_in_doc: int
    raw_text: str
    neighbour_terms: List[PydanticNeighbourTerm]
    

class PydanticDocument(BaseModel):
    doc_id: str
    title: str
    abstract: str
    weight: float
    neighbour_terms: List[PydanticNeighbourTerm]
    sentences: List[PydanticSentence]


class PydanticRanking(BaseModel):
    visible_neighbour_terms: List[PydanticNeighbourTerm]
    complete_neighbour_terms: List[PydanticNeighbourTerm]
    documents: List[PydanticDocument]