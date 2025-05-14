# import pandas as pd
# import requests
# import time
# from rdkit import Chem
# import logging
# import re

# logging.basicConfig(
#     filename='ligand_generator.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [%(lineno)d] %(message)s'
# )

# class LigandGenerator:
#     def __init__(self, api_key, model, system_file_path, preprompt_file_path, postprompt_file_path):
#         self.api_key = api_key
#         self.model = model
#         self.system = self._read_file(system_file_path)
#         self.preprompt = self._read_file(preprompt_file_path)
#         self.postprompt = self._read_file(postprompt_file_path)
#         self.headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }

#     def _read_file(self, file_path):
#         with open(file_path, 'r') as file:
#             return file.read().strip()

#     def generate_prompt(self, protein_sequence):
#         question = f"<question>\nProtein sequence: {protein_sequence}\n</question>"
#         logging.info(f'protein_sequence: {protein_sequence}')
#         return f"{self.system}\n{self.preprompt}\n{question}\n{self.postprompt}"

#     def send_request(self, prompt, max_retries=3, delay=5):
#         url = "https://api.vsegpt.ru/v1/chat/completions"

#         payload = {
#             "model": self.model,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 300
#         }

#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(url, headers=self.headers, json=payload, timeout=60)
#                 if response.status_code == 200:
#                     logging.info("Успешный ответ от API")
#                     return response.json()
#                 else:
#                     logging.warning(f"Ошибка API: {response.status_code} — {response.text}")
#                     time.sleep(delay)
#             except requests.exceptions.RequestException as e:
#                 logging.error(f"🌐 Сетевая ошибка: {e}")
#                 time.sleep(delay)

#         logging.error("Не удалось получить ответ после нескольких попыток.")
#         return None

#     def is_valid_smiles(self, smiles):
#         """Проверяет, является ли строка валидным SMILES."""
#         if not isinstance(smiles, str) or len(smiles.strip()) == 0:
#             return False
#         try:
#             mol = Chem.MolFromSmiles(smiles.strip())
#             return mol is not None
#         except:
#             return False

#     def extract_smiles(self, text):
#         """
#         Извлекает SMILES из текста модели.
#         Поддерживает форматы:
#         - `O=C(Nc1ccc(cc1)C(=O)O)c2cccnc2`
#         - ```smiles\n...\n```
#         """
#         if not isinstance(text, str):
#             return None

#         # Поиск через регулярные выражения

#         # Формат: `...`
#         backtick_match = re.search(r'`([^`\n]+)`', text)
#         if backtick_match:
#             candidate = backtick_match.group(1).strip()
#             if self.is_valid_smiles(candidate):
#                 return candidate

#         # Формат: ```smiles ... ```
#         code_block_match = re.search(r'```(?:smiles)?\s*([^\`]+)', text, re.DOTALL)
#         if code_block_match:
#             candidate = code_block_match.group(1).strip()
#             if self.is_valid_smiles(candidate):
#                 return candidate

#         # Формат: SMILES: ...
#         keywords = [
#             "SMILES:", "smiles:", "Correct molecular structure:",
#             "Molecular structure:", "Ligand SMILES:"
#         ]
#         for keyword in keywords:
#             if keyword in text:
#                 content_after = text.split(keyword, 1)[-1].strip()
#                 candidate = content_after.split("\n", 1)[0].strip()
#                 candidate = candidate.split(".", 1)[0].strip()
#                 if self.is_valid_smiles(candidate):
#                     return candidate

#         # Если ничего не нашлось
#         logging.warning("⚠️ SMILES не найден в ответе.")
#         return None

#     def process_dataframe(self, df, target_column="Target", output_column="Predicted_Label"):
#         generated_smiles = []

#         for idx, row in df.iterrows():
#             protein_sequence = row[target_column]
#             prompt = self.generate_prompt(protein_sequence)
#             logging.info(f"[{idx}] Prompt: {prompt[:200]}...")

#             try:
#                 response = self.send_request(prompt)
#                 if response is None:
#                     logging.warning(f"[{idx}] Ответ от модели пустой.")
#                     generated_smiles.append(None)
#                     continue

#                 content = response["choices"][0]["message"]["content"]
#                 logging.info(f"[{idx}] Raw Response: {content}")

#                 smiles = self.extract_smiles(content)
#                 if smiles and self.is_valid_smiles(smiles):
#                     logging.info(f"[{idx}] ✅ Валидный SMILES: {smiles}")
#                     generated_smiles.append(smiles)
#                 else:
#                     logging.warning(f"[{idx}] Невалидный или отсутствующий SMILES.")
#                     generated_smiles.append(None)

#             except Exception as e:
#                 logging.error(f"[{idx}] Ошибка: {str(e)}")
#                 generated_smiles.append(None)

#             time.sleep(1)

#         df[output_column] = generated_smiles
#         return df

#     def save_results(self, df, output_file):
#         df.to_csv(output_file, index=False)
#         print(f"💾 Результаты сохранены в {output_file}")
#         logging.info(f"Результаты сохранены в {output_file}")
##############
import pandas as pd
import requests
import time
from rdkit import Chem
import logging
import re
import numpy as np
from keras.models import load_model
from lifelines.utils import concordance_index as cindex_score
import os

# Настройки логирования
logging.basicConfig(
    filename='ligand_generator_deepseek_coder_2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(lineno)d] %(message)s'
)

class LigandGeneratorWithAffinityControl:
    def __init__(self, api_key, model_id, system_path, preprompt_path, postprompt_path, dta_model_path):
        """
        :param api_key: VseGPT API ключ
        :param model_id: ID модели на VseGPT (например, mistralai/mistral-large-2411)
        :param system_path: system.txt
        :param preprompt_path: preprompt.txt
        :param postprompt_path: document.txt
        :param dta_model_path: Путь к предобученной DTA-модели (например, combined_davis.h5)
        """
        self.api_key = api_key
        self.model_id = model_id
        self.system = self._read_file(system_path)
        self.preprompt = self._read_file(preprompt_path)
        self.postprompt = self._read_file(postprompt_path)
        self.dta_model = self.load_dta_model(dta_model_path)

        # Словари токенизации из оригинального датасета DeepDTA
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
        """Чтение файла с промтами"""
        with open(file_path, 'r') as f:
            return f.read().strip()

    def generate_prompt(self, protein_sequence):
        """Генерация запроса для LLM"""
        question = f"<question>\nProtein sequence: {protein_sequence}\n</question>"
        return f"{self.system}\n{self.preprompt}\n{question}\n{self.postprompt}"

    def send_request(self, prompt, max_retries=3, delay=5):
        """Отправка запроса к API VseGPT"""
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
                    logging.info("✅ Успешный ответ от API")
                    return response.json()
                else:
                    logging.warning(f"❌ Ошибка API: {response.status_code} — {response.text}")
                    time.sleep(delay)
            except Exception as e:
                logging.error(f"🌐 Сетевая ошибка: {e}")
                time.sleep(delay)

        logging.error("🚫 Не удалось получить ответ после нескольких попыток.")
        return None

    def is_valid_smiles(self, smiles):
        """Проверяет валидность SMILES через RDKit"""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def extract_smiles(self, text):
        """Извлечение SMILES из текстового ответа"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return None

        keywords = [
            "Correct molecular structure:", "SMILES:", "Ligand SMILES:",
            "Generated SMILES:", "Molecular structure:"
        ]

        for keyword in keywords:
            if keyword in text:
                content_after = text.split(keyword, 1)[-1].strip()
                candidate = content_after.split("\n", 1)[0].strip()
                candidate = candidate.split(".", 1)[0].strip()
                if self.is_valid_smiles(candidate):
                    return candidate

        backtick_match = re.search(r'`([^`\n]+)`', text)
        if backtick_match:
            candidate = backtick_match.group(1).strip()
            if self.is_valid_smiles(candidate):
                return candidate

        code_block_match = re.search(r'```(?:smiles)?\s*([^\`]+)', text, re.DOTALL)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            candidate = candidate.split("\n")[0]
            if self.is_valid_smiles(candidate):
                return candidate

        logging.warning("⚠️ SMILES не найден в ответе.")
        return None

    def encode_smiles(self, smiles, max_len=100):
        """Кодирование SMILES в числовое представление"""
        encoded = np.zeros((1, max_len))
        for i, ch in enumerate(smiles[:max_len]):
            encoded[0, i] = self.smiles_dict.get(ch, 0)
        return encoded

    def encode_protein(self, sequence, max_len=1000):
        """Кодирование белковой последовательности"""
        encoded = np.zeros((1, max_len))
        for i, aa in enumerate(sequence[:max_len]):
            encoded[0, i] = self.protein_dict.get(aa, 0)
        return encoded

    def load_dta_model(self, path):
        """Загрузка предобученной DTA-модели"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель не найдена: {path}")

        return load_model(path, custom_objects={'cindex_score': lambda y_true, y_pred: 0})

    def predict_affinity(self, protein_seq, smiles):
        """Предсказание аффинности связывания"""
        if not protein_seq or not smiles:
            return None

        X_drug = self.encode_smiles(smiles)
        X_target = self.encode_protein(protein_seq)

        try:
            affinity = self.dta_model.predict([X_drug, X_target], verbose=0)[0][0]
            return float(affinity)
        except Exception as e:
            logging.error(f"Ошибка предсказания аффинности: {str(e)}")
            return None

    def process_dataframe(self, df, target_column="Target", output_column="Generated_SMILES", min_affinity=7.0):
        """Обработка всего DataFrame с перегенерацией при низкой аффинности"""
        results = []

        for idx, row in df.iterrows():
            protein_sequence = row[target_column]
            prompt = self.generate_prompt(protein_sequence)
            logging.info(f"[{idx}] Prompt: {prompt[:200]}...")

            best_smiles = None
            attempts = 0

            while attempts < 3 and best_smiles is None:
                try:
                    response = self.send_request(prompt)
                    if response is None:
                        prompt += "\nAPI вернул ошибку. Перегенерирую."
                        attempts += 1
                        continue

                    content = response["choices"][0]["message"]["content"]
                    logging.info(f"[{idx}] Raw Response: {content}")

                    smiles = self.extract_smiles(content)

                    if not smiles or not self.is_valid_smiles(smiles):
                        prompt += "\nСгенерированный SMILES невалиден. Генерирую новый..."
                        attempts += 1
                        continue

                    # Предсказание аффинности
                    affinity = self.predict_affinity(protein_sequence, smiles)

                    if affinity is None:
                        prompt += "\nНе удалось рассчитать аффинность. Генерирую новый лиганд..."
                        attempts += 1
                        continue

                    logging.info(f"[{idx}] Предсказанная аффинность: {affinity:.2f}")

                    if affinity >= min_affinity:
                        logging.info(f"[{idx}] ✅ Найден лиганд с высокой аффинностью: {smiles}")
                        results.append({
                            "Sequence": protein_sequence,
                            "Predicted_Label": smiles,
                            "Predicted_Affinity": affinity
                        })
                    else:
                        prompt += f"\nPredicted affinity: {affinity:.2f}. Это ниже порога {min_affinity}. Генерирую новый лиганд с улучшенным связыванием."
                        attempts += 1

                except Exception as e:
                    logging.error(f"[{idx}] 💥 Ошибка: {str(e)}")
                    prompt += "\nПроизошла ошибка. Попробую ещё раз."
                    attempts += 1
                    time.sleep(2)

            if best_smiles is None:
                results.append({
                    "Sequence": protein_sequence,
                    "Predicted_Label": None,
                    "Predicted_Affinity": None
                })

            time.sleep(1)

        result_df = pd.DataFrame(results)
        return result_df

    def save_results(self, df, output_path):
        """Сохранение результата в CSV"""
        df.to_csv(output_path, index=False)
        print(f"✅ Результаты сохранены в {output_path}")