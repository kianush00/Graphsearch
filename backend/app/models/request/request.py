from pydantic import BaseModel


class QueryTerm(BaseModel):
    query: str 
