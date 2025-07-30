from pydantic import BaseModel
from typing import List, Tuple


class RerankNewPositions(BaseModel):
    ranking_new_positions: list[int]
    ranking_excluded_documents: list[int]


class PydanticNeighbourTerm(BaseModel):
    term: str
    proximity_score: float
    frequency_score: float
    criteria: str

    
class PydanticSentence(BaseModel):
    position_in_doc: int
    raw_text: str
    raw_to_processed_map: List[Tuple[int, int, str, str]]
    all_neighbour_terms: List[PydanticNeighbourTerm]
    

class PydanticDocument(BaseModel):
    doc_id: str
    title: str
    abstract: str
    initial_position: int
    authors: str
    content_type: str
    publication_year: int
    citing_paper_count: int
    preprocessed_text: str
    weight: float
    all_neighbour_terms: List[PydanticNeighbourTerm]
    sentences: List[PydanticSentence]


class PydanticRanking(BaseModel):
    visible_neighbour_terms: List[PydanticNeighbourTerm]
    complete_neighbour_terms: List[PydanticNeighbourTerm]
    documents: List[PydanticDocument]
    individual_query_terms_list: List[str]