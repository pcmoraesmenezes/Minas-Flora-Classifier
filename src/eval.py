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
        
        self.processor, self.model = self._load_model()
        logging.info(f"Processor and model loaded from {self.model_path}")
        
        self.dataset, self.label_map = self._load_eval_dataset()
        
        self.metrics = {class_name: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for class_name in self.dataset.keys()}
        logging.info(f'Metrics initialized for classes: {list(self.metrics.keys())}')
        
        self._evaluate()
        
        for class_name in self.dataset.keys():
            print(f"\n--- Metrics for: {class_name} ---")
            self._precision(class_name)
            self._recall(class_name)
            self._f1_score(class_name)

    def _load_model(self):
        model = CLIPModel.from_pretrained(self.model_path)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return processor, model
    
    def __read_labels(self, label_path):
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} does not exist.")
        with open(label_path, 'r') as f:
            prompt = f.read().strip()
        return prompt

    def _get_class_name_from_path(self, label_path):
        basename = os.path.basename(label_path)
        class_name = os.path.splitext(basename)[0]
        return class_name.replace('-', ' ').capitalize()

    def _load_eval_dataset(self):
        if not os.path.exists(self.eval_path):
            raise FileNotFoundError(f"Evaluation path {self.eval_path} does not exist.")
        
        dataset = {}
        label_map = {}
        
        for dir_name in os.listdir(self.eval_path):
            dir_path = os.path.join(self.eval_path, dir_name)
            if not os.path.isdir(dir_path):
                continue

            logging.info(f"Processing directory: {dir_name}")
            
            files_in_subdir = os.listdir(dir_path)
            label_file = next((f for f in files_in_subdir if f.endswith('.txt')), None)

            if label_file:
                label_path = os.path.join(dir_path, label_file)
                prompt = self.__read_labels(label_path)
                class_name = self._get_class_name_from_path(label_path)
                
                logging.info(f"Directory '{dir_name}' -> Class: '{class_name}'")
                
                label_map[class_name] = prompt
                
                images = [os.path.join(dir_path, file) for file in files_in_subdir if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                if images:
                    dataset[class_name] = images
                    logging.info(f"Added {len(images)} images for class '{class_name}'.")
            else:
                logging.warning(f"No label file found in directory {dir_name}. Skipping.")
        
        if not dataset:
            raise ValueError("No images found in the evaluation path.")
        return dataset, label_map


    def _get_winner_index(self, x, y):
        cosine_similarity = (x @ y.T) / (torch.linalg.norm(x) * torch.linalg.norm(y, dim=1))
        return torch.argmax(cosine_similarity)

    
    def _predict(self, image, all_prompts):
        inputs = self.processor(text=list(all_prompts), images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            
            outputs = self.model(**inputs)
            winner_idx = self._get_winner_index(outputs.image_embeds, outputs.text_embeds)
            winner_label = all_prompts[winner_idx.item()]
            logging.info(f"Classified image with label: {winner_label}")

            
            prompt_to_class_map = {v: k for k, v in self.label_map.items()}
            predicted_class_name = prompt_to_class_map[winner_label]

            logging.info(f"Predicted class: '{predicted_class_name}'")
            return winner_label

    def _evaluate(self):
        all_class_names = list(self.dataset.keys())
        all_prompts = list(self.label_map.values())
        prompt_to_class_map = {v: k for k, v in self.label_map.items()}
        
        logging.info(f'All available classes: {all_class_names}')
        total_samples = sum(len(paths) for paths in self.dataset.values())
        
        for true_class_name in all_class_names:
            images_paths = self.dataset.get(true_class_name, [])
            logging.info(f'Evaluating {len(images_paths)} images for the class: {true_class_name}')
            
            for image_path in images_paths:
                logging.info(f"-> Processing image: {image_path}")
                try:
                    img = Image.open(image_path).convert("RGB")
                except Exception as e:
                    logging.error(f"Could not open image {image_path}: {e}")
                    continue
                
                predicted_prompt = self._predict(img, all_prompts)
                predicted_class_name = prompt_to_class_map[predicted_prompt]
                
                if predicted_class_name == true_class_name:
                    self.metrics[true_class_name]["TP"] += 1
                else:
                    self.metrics[predicted_class_name]["FP"] += 1
                    self.metrics[true_class_name]["FN"] += 1
        
        logging.info(f"Completed primary evaluation loop. Total samples: {total_samples}")
        
        logging.info("Calculating True Negatives (TN)...")
        for class_name in all_class_names:
            tp = self.metrics[class_name]["TP"]
            fp = self.metrics[class_name]["FP"]
            fn = self.metrics[class_name]["FN"]
            self.metrics[class_name]["TN"] = total_samples - (tp + fp + fn)
            
        logging.info("--- EVALUATION COMPLETE ---")
        for class_name, metric_values in self.metrics.items():
            logging.info(f"Raw Metrics for '{class_name}': {metric_values}")
            
    def _recall(self, class_name):
        tp = self.metrics[class_name]["TP"]
        fn = self.metrics[class_name]["FN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        logging.info(f"Recall for {class_name}: {recall:.4f}")
        return recall

    def _precision(self, class_name):
        tp = self.metrics[class_name]["TP"]
        fp = self.metrics[class_name]["FP"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        logging.info(f"Precision for {class_name}: {precision:.4f}")
        return precision

    def _f1_score(self, class_name):
        precision = self._precision(class_name)
        recall = self._recall(class_name)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        logging.info(f"F1 Score for {class_name}: {f1_score:.4f}")
        return f1_score

if __name__ == "__main__":
    model_path = 'minas_flora_classifier_model'
    eval_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    evaluator = MinasFloraEvaluation(model_path, eval_path)
    logging.info("Evaluation process finished.")