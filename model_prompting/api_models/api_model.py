# 

import pandas as pd
import numpy as np
import time
import re
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
from keras.models import load_model
from lifelines.utils import concordance_index as cindex_score
import os

# Настройка логирования
logging.basicConfig(
    filename='ligand_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(lineno)d] %(message)s'
)

class LigandGenerator:
    def __init__(self, api_key, model_id, system_path, preprompt_path, dta_model_path):
        self.api_key = api_key
        self.model_id = model_id
        self.system = self._read_file(system_path)
        self.preprompt = self._read_file(preprompt_path)
        self.dta_model = self.load_dta_model(dta_model_path)

        # Словари токенизации
        self.smiles_dict = {
            '': 0, '#': 1, '%': 2, ')': 3, '(': 4,
            '+': 5, '-': 6, '.': 7, '0': 8, '1': 9,
            '2': 10, '3': 11, '4': 12, '5': 13, '6': 14,
            '7': 15, '8': 16, '9': 17, '=': 18, '@': 19,
            'A': 20, 'B': 21, 'C': 22, 'F': 23, 'H': 24,
            'I': 25, 'N': 26, 'O': 27, 'P': 28, 'S': 29,
            '[': 30, '\\': 31, ']': 32, '_': 33, 'a': 34,
            'c': 35, 'e': 36, 'g': 37, 'i': 38, 'l': 39,
            'n': 40, 'o': 41, 'r': 42, 's': 43, 't': 44,
            'u': 45, '|': 46
        }

        self.protein_dict = {
            '': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
            'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14,
            'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19,
            'Y': 20
        }

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read().strip()

    def generate_prompt(self, protein_sequence):
        question = f"<question>\nProtein sequence: {protein_sequence}\n</question>"
        answer_format = "Please respond ONLY in the following format:\nCorrect molecular structure: <SMILES>"

        return f"{self.system}\n{self.preprompt}\n{question}\n{answer_format}"

    def send_request(self, prompt, max_retries=3, delay=5):
        url = "https://api.vsegpt.ru/v1/chat/completions"

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=60)
                if response.status_code == 200:
                    logging.info("Успешный ответ от API")
                    return response.json()
                else:
                    logging.warning(f"Ошибка API: {response.status_code} — {response.text}")
                    time.sleep(delay)
            except Exception as e:
                logging.error(f"🌐 Сетевая ошибка: {e}")
                time.sleep(delay)

        logging.error("🚫 Не удалось получить ответ после нескольких попыток.")
        return None

    def is_valid_smiles(self, smiles):
        """Проверяет, является ли строка валидным SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            return mol is not None
        except:
            return False

    def extract_smiles(self, text):
        """Извлечение SMILES из текстового ответа модели"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return None

        # По ключевому слову
        if "Correct molecular structure:" in text:
            content_after = text.split("Correct molecular structure:")[1].split("\n")[0].strip()
            if self.is_valid_smiles(content_after):
                return content_after

        # По коду в кавычках
        backtick_match = re.search(r'`([^`\n]+)`', text)
        if backtick_match:
            candidate = backtick_match.group(1).strip()
            if self.is_valid_smiles(candidate):
                return candidate

        # По код-блокам
        code_block_match = re.search(r'```(?:smiles)?\s*([^\`]+)', text, re.DOTALL)
        if code_block_match:
            candidate = code_block_match.group(1).strip().split('\n')[0]
            if self.is_valid_smiles(candidate):
                return candidate

        # По любому SMILES в теле ответа
        simple_match = re.search(r'(?:CC|NC|O=C|C1=CC=).*?(?=\s|$)', text)
        if simple_match:
            candidate = simple_match.group(0).strip()
            if self.is_valid_smiles(candidate):
                return candidate

        logging.warning("SMILES не найден в ответе.")
        return None

    def encode_smiles(self, smiles, max_len=100):
        encoded = np.zeros((1, max_len))
        for i, ch in enumerate(smiles[:max_len]):
            if ch in self.smiles_dict:
                encoded[0, i] = self.smiles_dict[ch]
        return encoded

    def encode_protein(self, sequence, max_len=1000):
        encoded = np.zeros((1, max_len))
        for i, aa in enumerate(sequence[:max_len]):
            if aa in self.protein_dict:
                encoded[0, i] = self.protein_dict[aa]
        return encoded

    def load_dta_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель не найдена: {path}")
        return load_model(path, custom_objects={'cindex_score': lambda y_true, y_pred: 0})

    def predict_affinity(self, protein_seq, smiles):
        if not protein_seq or not smiles:
            return None

        X_drug = self.encode_smiles(smiles)
        X_target = self.encode_protein(protein_seq)

        try:
            affinity = self.dta_model.predict([X_drug, X_target], verbose=0)[0][0]
            return float(affinity)
        except Exception as e:
            logging.error(f"Ошибка предсказания аффинности: {e}")
            return None

    def process_dataframe(self, df, target_column="Target", output_column="Generated_SMILES", min_affinity=0.6, max_attempts=5):
        results = []

        for idx, row in df.iterrows():
            protein_sequence = row[target_column]
            prompt_base = self.generate_prompt(protein_sequence)

            best_smiles = None
            attempts = 0

            while attempts < max_attempts and best_smiles is None:
                prompt = prompt_base + (
                    "\nGenerate a different ligand with higher binding potential." if attempts > 0 else ""
                )

                try:
                    response = self.send_request(prompt)
                    if response is None:
                        attempts += 1
                        continue

                    content = response["choices"][0]["message"]["content"]
                    logging.info(f"[{idx}] Raw Response: {content}")

                    smiles = self.extract_smiles(content)

                    if not smiles or not self.is_valid_smiles(smiles):
                        logging.warning(f"[{idx}] Невалидный или отсутствующий SMILES.")
                        attempts += 1
                        continue

                    # Предсказание аффинности
                    affinity = self.predict_affinity(protein_sequence, smiles)

                    if affinity is None:
                        logging.warning(f"[{idx}] Не удалось рассчитать аффинность.")
                        attempts += 1
                        continue

                    logging.info(f"[{idx}] Предсказанная аффинность: {affinity:.2f}")

                    if affinity >= min_affinity:
                        logging.info(f"[{idx}] Найден подходящий лиганд: {smiles}")
                        best_smiles = smiles
                    else:
                        prompt_base += f"\nPredicted affinity: {affinity:.2f}. Это ниже порога {min_affinity}. Генерирую ещё раз..."
                        attempts += 1

                except Exception as e:
                    logging.error(f"[{idx}] Ошибка при обработке запроса: {str(e)}")
                    attempts += 1
                    time.sleep(1)

            if best_smiles:
                results.append({
                    "Sequence": protein_sequence,
                    "Generated_SMILES": best_smiles,
                    "Predicted_Affinity": self.predict_affinity(protein_sequence, best_smiles),
                    "Attempts": attempts
                })
            else:
                results.append({
                    "Sequence": protein_sequence,
                    "Generated_SMILES": None,
                    "Predicted_Affinity": None,
                    "Attempts": max_attempts
                })

            time.sleep(1)

        result_df = pd.DataFrame(results)
        return result_df

    def save_results(self, df, output_file):
        df.to_csv(output_file, index=False)
        print(f"Результаты сохранены в {output_file}")