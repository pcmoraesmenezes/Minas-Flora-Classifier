import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MinasFloraDataset():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        self._get_dataset()
        logging.info(f"Dataset initialized with {len(self.samples)} samples.")
    
    def _get_dataset(self):
        for root,dirs,files in os.walk(self.root_dir):
            for file in files:
                logging.info(f"Processing file: {file}")
                img_path = None
                label_path = None

                if file.endswith('.jpg') or file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    label_file = file.rsplit('.', 1)[0] + '.txt'
                    label_path = os.path.join(root, label_file)
                    
                if img_path and os.path.exists(label_path):
                    self.samples.append((img_path, label_path))
                    logging.info(f"Added sample: {img_path} with label: {label_path}")

    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of range")
        
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path)
        with open(label_path, 'r') as f:
            label = f.read().strip()
            return image, label
        
        
if __name__ == "__main__":
    dataset = MinasFloraDataset('/home/paulo/Área de Trabalho/repos/Minas-Flora-Classifier/data')
    print(f"Total samples in dataset: {len(dataset)}")
    img, label = dataset[0]
    print(f"First sample - Image size: {img.size}, Label: {label}")
    
