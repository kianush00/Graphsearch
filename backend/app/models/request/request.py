from pydantic import BaseModel
from typing import List


class QueryTerm(BaseModel):
    query: str 


class PydanticNeighboursTerm(BaseModel):
    term: str
    frequency: float
    distance: float


class PydanticNeighboursTerms(BaseModel):
    neighbour_terms: List[PydanticNeighboursTerm]