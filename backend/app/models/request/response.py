from pydantic import BaseModel
from typing import List


class RerankNewPositions(BaseModel):
    ranking_new_positions: list[int]

class PydanticNeighbourTerm(BaseModel):
    term: str
    ponderation: float
    distance: float
    

class PydanticDocument(BaseModel):
    doc_id: str
    title: str
    abstract: str
    neighbour_terms: List[PydanticNeighbourTerm]


class PydanticRanking(BaseModel):
    visible_neighbour_terms: List[PydanticNeighbourTerm]
    complete_neighbour_terms: List[PydanticNeighbourTerm]
    documents: List[PydanticDocument]