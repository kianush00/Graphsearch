from services.services import queryService
from fastapi import APIRouter
from models.request.request import QueryTerm
from models.request.response import PydanticRanking

router = APIRouter()


@router.post("/get-ranking")
async def get_neighbour_terms(queryTerm: QueryTerm) -> PydanticRanking:
    return queryService.processQuery(queryTerm.query)
        