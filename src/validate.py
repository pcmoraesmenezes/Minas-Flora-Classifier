from data_classifier import MinasFloraClassifier
from data_loader import MinasFloraDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_model(dataset_path, model_name='openai/clip-vit-base-patch32'):
    dataset = MinasFloraDataset(dataset_path)
    classifier = MinasFloraClassifier(model_name)
    
    correct_predictions = 0
    total_samples = len(dataset)
    
    logging.info(f"Starting validation on {total_samples} samples.")
    
    all_labels_path = list(set(label for _, label in dataset.samples))
    logging.info(f"Unique labels found: {all_labels_path}")
    
    all_labels = []
    
    for label_path in all_labels_path:
        with open(label_path, 'r') as f:
            label = f.read().strip()
            all_labels.append(label)
            
    logging.info(f"All labels for classification: {all_labels}")
    
    
    for idx in range(total_samples):
        image, label = dataset[idx]
        predicted_label = classifier.classify(image, all_labels)

        if predicted_label == label:
            correct_predictions += 1
            logging.info(f"Sample {idx}: Correctly classified as {predicted_label}.")
        else:
            logging.warning(f"Sample {idx}: Classified as {predicted_label}, but actual label is {label}.")
    
    accuracy = correct_predictions / total_samples * 100
    logging.info(f"Validation completed. Accuracy: {accuracy:.2f}%")
    return accuracy





if __name__ == "__main__":
    dataset_path = '/home/paulo/Área de Trabalho/repos/Minas-Flora-Classifier/data'
    model_name = 'openai/clip-vit-base-patch32'
    validate_model(dataset_path, model_name)