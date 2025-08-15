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
        
        
    def _get_winner_index(self, x, y):
        return torch.argmax((x @ y) / (torch.linalg.norm(x) * torch.linalg.norm(y)), dim=-1)
    
    
    def classify(self, image, all_labels):
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")
        
        inputs = self.processor(text=all_labels, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        
        winner_idx = self._get_winner_index(outputs.image_embeds, outputs.text_embeds)
        winner_label = all_labels[winner_idx.item()]
        logging.info(f"Classified image with label: {winner_label}")
        return winner_label