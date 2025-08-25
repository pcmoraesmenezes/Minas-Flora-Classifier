from fastapi import FastAPI
from src.app.api.app import api_router, lifespan 

app = FastAPI(
    title="Minas Flora Classifier API",
    description="API para classificar imagens de plantas da flora de Minas Gerais.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api")

@app.get("/", tags=["Root"], summary="Verifica a saúde da API")
def read_root():
    """Endpoint raiz para verificar se a API está no ar."""
    return {"status": "ok", "message": "Serviço Minas Flora Classifier está no ar!"}
