import logging
import torch
from transformers import CLIPModel, AutoProcessor
import os
from PIL import Image


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MinasFloraEvaluation:
    def __init__(self, model_path, eval_path):
        self.model_path = model_path
        self.eval_path = eval_path
        
        self.model, self.processor = self._load_model()
        logging.info(f"Model loaded from {self.model_path}")
        
        self.dataset = self._load_eval_dataset()
        
        self.metrics = {label: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for label in self.dataset.keys()}
        logging.info(f'Metrics initialized {self.metrics}')

    def _load_model(self):
        model = CLIPModel.from_pretrained(self.model_path)
        processor = AutoProcessor.from_pretrained(self.model_path)
        logging.info(f"Processor loaded from {self.model_path}")
        return model, processor
    

    def _load_eval_dataset(self):
        if not os.path.exists(self.eval_path):
            raise FileNotFoundError(f"Evaluation path {self.eval_path} does not exist.")
        
        dataset = {}
        for root, dirs, files in os.walk(self.eval_path):
            for dir_name in dirs:
                logging.info(f"Processing directory: {dir_name}")
                dir_path = os.path.join(root, dir_name)
                logging.info(f'Directory path: {dir_path}')
                images = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(('.jpg', '.png'))]
                logging.info(f"Found {len(images)} images in {dir_name}.")
                if images:
                    dataset[dir_name] = images
                    logging.info(f"Found {len(images)} images in class {dir_name}.")
                    logging.info(f"Images: {images}")
        if not dataset:
            raise ValueError("No images found in the evaluation path.")
        return dataset
    
    
    def _predict(self, image, all_labels):
        inputs = self.processor(text=list(all_labels), images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            predicted_label_idx = probs.argmax().item()
            predicted_label = list(all_labels)[predicted_label_idx]
            logging.info(f"Predicted label: {predicted_label} with probability {probs[0][predicted_label_idx]:.4f}")
            return predicted_label
    
    
    def evaluate(self):
        all_labels = self.dataset.keys()
        logging.info(f'All labels avaliable: {all_labels}')
        
        for label in all_labels:
            images = self.dataset.get(label)
            logging.info(f'images: {images}')
            
            for image in images:
                logging.info(f"Evaluating image: {image} for label: {label}")
                img = Image.open(image)
                
                predicted_label = self._predict(img, all_labels)
                logging.info(f"Image {image} predicted as {predicted_label}")
                if predicted_label == label:
                    self.metrics[label]["TP"] += 1
                    logging.info(f"True Positive for label {label}.")
                else:
                    self.metrics[label]["FP"] += 1
                    logging.info(f"False Positive for label {label}.")
                for other_label in all_labels:
                    if other_label != label:
                        if predicted_label == other_label:
                            self.metrics[other_label]["FN"] += 1
                            logging.info(f"False Negative for label {other_label}.")
                        else:
                            self.metrics[other_label]["TN"] += 1
                            logging.info(f"True Negative for label {other_label}.")
                logging.info(f"Metrics for label {label}: {self.metrics[label]}")

            


if __name__ == "__main__":
    
    model_path = 'minas_flora_classifier_model'
    eval_path = '/home/paulo/Área de Trabalho/repos/Minas-Flora-Classifier/validate-data'
    
    evaluator = MinasFloraEvaluation(model_path, eval_path)
    evaluator.evaluate()
    logging.info("Evaluation setup complete.")
    
