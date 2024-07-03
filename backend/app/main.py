from app.services.services import queryService
from app.models.request.request import QueryTerm, PydanticNeighboursTerms
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()

origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/get-neighbour-terms")
async def get_neighbour_terms(queryTerm: QueryTerm) -> PydanticNeighboursTerms:
    return queryService.processQuery(queryTerm.query)




if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8080, reload=True)