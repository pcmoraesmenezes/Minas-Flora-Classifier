import os
from PIL import Image
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MinasFloraProcessor:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = prompt_dir
        self.prompts: Dict[str, str] = {}

        self._load_prompts()
        logging.info(f"Processador inicializado com {len(self.prompts)} prompts.")

    def _load_prompts(self):
        logging.info(f"Carregando prompts do diretório: {self.prompt_dir}")
        if not os.path.isdir(self.prompt_dir):
            logging.error(f"O diretório de prompts não foi encontrado: {self.prompt_dir}")
            raise FileNotFoundError(f"O diretório de prompts não foi encontrado: {self.prompt_dir}")

        for file_name in os.listdir(self.prompt_dir):
            if file_name.endswith('.txt'):
                class_label = os.path.splitext(file_name)[0]
                prompt_path = os.path.join(self.prompt_dir, file_name)
                try:
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt_text = f.read().strip()
                        self.prompts[class_label] = prompt_text
                        logging.info(f"Prompt carregado para a classe '{class_label}'")
                except Exception as e:
                    logging.error(f"Falha ao ler o arquivo de prompt {prompt_path}: {e}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __call__(self, image: Image.Image) -> Tuple[Image.Image, List[str], List[str]]:
        if not self.prompts:
            logging.warning("Nenhum prompt foi carregado. O processamento pode não funcionar como esperado.")
            return image, [], []

        labels = list(self.prompts.keys())
        prompt_texts = list(self.prompts.values())
        
        return image, labels, prompt_texts