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
        
        self._evaluate()
        
        for label in self.dataset.keys():
            self._recall(label)
            self._precision(label)
            self._f1_score(label)

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
    
    
    def _evaluate(self):
        all_labels = list(self.dataset.keys())
        logging.info(f'All available labels: {all_labels}')
        
        total_samples = 0
        for true_label in all_labels:
            images_paths = self.dataset.get(true_label, [])
            total_samples += len(images_paths) 
            
            logging.info(f'Evaluating {len(images_paths)} images for the class: {true_label}')
            
            for image_path in images_paths:
                logging.info(f"-> Processing image: {image_path}")
                img = Image.open(image_path)
                
                predicted_label = self._predict(img, all_labels)
                
                if predicted_label == true_label:
                    self.metrics[true_label]["TP"] += 1
                else:
                    self.metrics[predicted_label]["FP"] += 1
                    self.metrics[true_label]["FN"] += 1
        
        logging.info(f"Completed primary evaluation loop. Total samples: {total_samples}")
        
        logging.info("Calculating True Negatives (TN) by derivation...")
        for label in all_labels:
            tp = self.metrics[label]["TP"]
            fp = self.metrics[label]["FP"]
            fn = self.metrics[label]["FN"]
            
            tn = total_samples - (tp + fp + fn)
            self.metrics[label]["TN"] = tn
            
        logging.info("--- EVALUATION COMPLETE ---")
        for label, metric_values in self.metrics.items():
            logging.info(f"Final Metrics for '{label}': {metric_values}")
            
            
    def _recall(self, label):
        tp = self.metrics[label]["TP"]
        fn = self.metrics[label]["FN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        logging.info(f"Recall for {label}: {recall:.4f}")
        return recall

    def _precision(self, label):
        tp = self.metrics[label]["TP"]
        fp = self.metrics[label]["FP"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        logging.info(f"Precision for {label}: {precision:.4f}")
        return precision

    def _f1_score(self, label):
        precision = self._precision(label)
        recall = self._recall(label)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        logging.info(f"F1 Score for {label}: {f1_score:.4f}")
        return f1_score




if __name__ == "__main__":
    
    model_path = 'minas_flora_classifier_model'
    eval_path = '/home/paulo/Área de Trabalho/repos/Minas-Flora-Classifier/validate-data'
    
    evaluator = MinasFloraEvaluation(model_path, eval_path)
    logging.info("Evaluation setup complete.")
    
