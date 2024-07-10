from services.services import queryService
from fastapi import APIRouter
from models.request.request import QueryTerm, PydanticRankingRequest
from models.request.response import PydanticRanking, RerankNewPositions

router = APIRouter()


@router.post("/get-ranking")
async def get_neighbour_terms(query_term: QueryTerm) -> PydanticRanking:
    return queryService.process_query(query_term.query)


@router.post("/rerank")
async def get_new_ranking_order(ranking: PydanticRankingRequest) -> RerankNewPositions:
    return queryService.get_new_ranking_positions(ranking)
        