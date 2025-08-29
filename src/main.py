from fastapi import FastAPI
from src.app.app import api_router, lifespan


app = FastAPI(
    title="Minas Flora Classifier API",
    description="API to classify plant images from the flora of Minas Gerais.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api")

