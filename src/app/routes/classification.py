import logging
from io import BytesIO
from PIL import Image

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

router = APIRouter()

@router.post("/classify", summary="Classifica uma imagem de planta")
async def classify_image(request: Request, image: UploadFile = File(..., description="Arquivo de imagem para classificação.")):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem.")

    processor = request.app.state.processor
    classifier = request.app.state.classifier

    logging.info(f"Received image: {image.filename} ({image.content_type})")

    try:
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents))
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=422, detail="Could not process image file.")

    _, _, prompt_texts = processor(pil_image) 
    
    predicted_prompt = classifier.classify(pil_image, prompt_texts)
    
    predicted_label = "unknown"
    for label, prompt_text in processor.prompts.items():
        if prompt_text == predicted_prompt:
            predicted_label = label
            break

    logging.info(f"Final label returned: {predicted_label}")

    return {"predicted_label": predicted_label}