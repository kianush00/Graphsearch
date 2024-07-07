from services.services import queryService
from fastapi import APIRouter
from models.request.request import QueryTerm, PydanticNeighboursTerms

router = APIRouter()


@router.post("/get-neighbour-terms")
async def get_neighbour_terms(queryTerm: QueryTerm) -> PydanticNeighboursTerms:
    return queryService.processQuery(queryTerm.query)