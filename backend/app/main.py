from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router
import logging
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

# App configuration
class Settings:
    PROJECT_NAME: str = "SREX Backend API"
    ALLOWED_ORIGINS: list = ["*"]
    ALLOW_METHODS: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOW_HEADERS: list = ["*"]
    ALLOW_CREDENTIALS: bool = False
    MAX_AGE: int = 3600

settings = Settings()

app = FastAPI(title=settings.PROJECT_NAME)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=settings.ALLOW_METHODS,
    allow_headers=settings.ALLOW_HEADERS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    max_age=settings.MAX_AGE,
)

# Global exception handling
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )

# Include main router
app.include_router(router)

@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "healthy"}
