import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter

from src.service.data_loader import MinasFloraInputHandler
from src.service.data_classifier import MinasFloraClassifier
from src.app.routes.classification import router as classification_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Carregando modelos e preparando a aplicação...")
    
    app.state.handler = MinasFloraInputHandler(prompt_dir="docs/prompt")
    app.state.classifier = MinasFloraClassifier(model_name="openai/clip-vit-base-patch32")
    
    logging.info("Modelos carregados com sucesso.")
    
    yield
    
    logging.info("Encerrando aplicação e limpando recursos.")
    app.state.handler = None
    app.state.classifier = None

api_router = APIRouter()

api_router.include_router(classification_router, prefix="/v1", tags=["Classification"])