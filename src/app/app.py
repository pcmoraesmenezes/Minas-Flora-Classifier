import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter

from .service.data_loader import MinasFloraProcessor
from .service.data_classifier import MinasFloraClassifier
from .routes.classification import router as classification_router
from .routes.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading models and preparing the application...")

    app.state.processor = MinasFloraProcessor(prompt_dir="docs/prompt")
    app.state.classifier = MinasFloraClassifier(model_name="openai/clip-vit-base-patch32")

    logging.info("Models loaded successfully.")
    
    yield
    
    logging.info("Shutting down application and cleaning up resources.")
    app.state.handler = None
    app.state.classifier = None

api_router = APIRouter()

api_router.include_router(classification_router, prefix="/v1", tags=["Classification"])
api_router.include_router(health_router, prefix="/v1", tags=["Health"])