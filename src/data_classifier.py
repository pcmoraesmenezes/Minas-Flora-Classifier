from PIL import Image
import logging
from transformers import AutoProcessor, CLIPModel
import torch


from .data_loader import MinasFloraDataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MinasFloraClassifier:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        logging.info(f"Model {model_name} loaded successfully.")
        
        
    def cosine_similarity(self, x, y):
        return torch.argmax((x @ y) / (torch.linalg.norm(x) * torch.linalg.norm(y)), dim=-1)