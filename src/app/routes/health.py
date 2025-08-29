import logging
from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["Root"], summary="Verifica a saúde da API")
def read_root():
    logging.info("Health check endpoint called.")
    return {"status": "ok", "message": "Minas Flora Classifier is online!"}
