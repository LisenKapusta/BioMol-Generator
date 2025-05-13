# import pandas as pd
# import requests
# import time

# import logging
# logging.basicConfig(filename='/Users/holodkovaelizavetaigorevna/graduate_work/BioMol-Generator/model_prompting/api_models/api.log', level=logging.INFO)

# class LigandGenerator:
#     def __init__(self, api_key, model, system_file_path, preprompt_file_path, postprompt_file_path):
#         """
#         Инициализация класса.
#         :param api_key: Ваш API-ключ для доступа к VseGPT.
#         :param model: ID модели, которую вы хотите использовать.
#         :param system_file_path: Путь к файлу system.txt.
#         :param preprompt_file_path: Путь к файлу preprompt.txt.
#         :param postprompt_file_path: Путь к файлу document.txt.
#         """
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
#         """Чтение содержимого файла."""
#         with open(file_path, 'r') as file:
#             return file.read()

#     def generate_prompt(self, protein_sequence):
#         """
#         Генерация полного промта для отправки в API.
#         :param protein_sequence: Белковая последовательность.
#         :return: Полный текст запроса.
#         """
#         question = f"<question>\nProtein sequence: {protein_sequence}\n</question>"
#         return f"{self.system}\n{self.preprompt}\n{question}\n{self.postprompt}"

#     def send_request(self, prompt):
#         """
#         Отправка запроса к API.
#         :param prompt: Текст запроса (промт).
#         :return: Ответ от API в виде JSON.
#         """
#         url = "https://api.vsegpt.ru/v1/chat/completions"
#         payload = {
#             "model": self.model,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 300
#         }
#         logging.info(f"payload: {payload}")

#         try:
#             response = requests.post(url, headers=self.headers, json=payload)
#             if response.status_code == 200:
#                 logging.info(f'response: {response.json()}')
#                 return response.json()
#             else:
#                 logging.info(f"Error: {response.status_code}, {response.text}")
#                 raise Exception(f"Error: {response.status_code}, {response.text}")
#         except Exception as e:
#             print(f"Request failed: {e}")
#             logging.info(f"Request failed: {e}")
#             return None

#     def process_dataframe(self, df, target_column="Target", output_column="Generated_SMILES"):
#         generated_smiles = []

#         for _, row in df.iterrows():
#             protein_sequence = row[target_column]
#             prompt = self.generate_prompt(protein_sequence)
#             try:
#                 response = self.send_request(prompt)
#                 if response is None:
#                     smiles = None
#                 else:
#                     result = response["choices"][0]["message"]["content"]
#                     print(f"API Response: {result}")  # Логирование ответа
#                     logging.info(f"API Response: {result}")
#                     if "Correct molecular structure:" in result:
#                         smiles = result.split("Correct molecular structure: ")[-1].split("\n")[0].strip()
#                         logging.info(f'correct molecular structer: {smiles}')
#                     else:
#                         smiles = None
#                         logging.info(f'non correct molecular structer: {smiles}')
#             except Exception as e:
#                 smiles = f"Error: {str(e)}"
#                 logging.info(f"Error: {str(e)}")
#             generated_smiles.append(smiles)

#             # Добавляем задержку, чтобы избежать ограничений API
#             time.sleep(1)

#         df[output_column] = generated_smiles
#         return df
#     def save_results(self, df, output_file):
#         """
#         Сохранение результатов в файл.
#         :param df: DataFrame с результатами.
#         :param output_file: Путь для сохранения файла.
#         """
#         df.to_csv(output_file, index=False)
#         print(f"Results saved to {output_file}")

import pandas as pd
import requests
import time
from rdkit import Chem
import logging
import re

# Настройка логирования
logging.basicConfig(
    filename='ligand_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(lineno)d] %(message)s'
)

class LigandGenerator:
    def __init__(self, api_key, model, system_file_path, preprompt_file_path, postprompt_file_path):
        self.api_key = api_key
        self.model = model
        self.system = self._read_file(system_file_path)
        self.preprompt = self._read_file(preprompt_file_path)
        self.postprompt = self._read_file(postprompt_file_path)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read().strip()

    def generate_prompt(self, protein_sequence):
        question = f"<question>\nProtein sequence: {protein_sequence}\n</question>"
        logging.info(f'protein_sequence: {protein_sequence}')
        return f"{self.system}\n{self.preprompt}\n{question}\n{self.postprompt}"

    def send_request(self, prompt, max_retries=3, delay=5):
        url = "https://api.vsegpt.ru/v1/chat/completions"

        payload = {
            "model": self.model,
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
            except requests.exceptions.RequestException as e:
                logging.error(f"🌐 Сетевая ошибка: {e}")
                time.sleep(delay)

        logging.error("🚫 Не удалось получить ответ после нескольких попыток.")
        return None

    def is_valid_smiles(self, smiles):
        """Проверяет, является ли строка валидным SMILES."""
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            return mol is not None
        except:
            return False

    def extract_smiles(self, text):
        """
        Извлекает SMILES из текста модели.
        Поддерживает форматы:
        - `O=C(Nc1ccc(cc1)C(=O)O)c2cccnc2`
        - ```smiles\n...\n```
        """
        if not isinstance(text, str):
            return None

        # Поиск через регулярные выражения

        # Формат: `...`
        backtick_match = re.search(r'`([^`\n]+)`', text)
        if backtick_match:
            candidate = backtick_match.group(1).strip()
            if self.is_valid_smiles(candidate):
                return candidate

        # Формат: ```smiles ... ```
        code_block_match = re.search(r'```(?:smiles)?\s*([^\`]+)', text, re.DOTALL)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            if self.is_valid_smiles(candidate):
                return candidate

        # Формат: SMILES: ...
        keywords = [
            "SMILES:", "smiles:", "Correct molecular structure:",
            "Molecular structure:", "Ligand SMILES:"
        ]
        for keyword in keywords:
            if keyword in text:
                content_after = text.split(keyword, 1)[-1].strip()
                candidate = content_after.split("\n", 1)[0].strip()
                candidate = candidate.split(".", 1)[0].strip()
                if self.is_valid_smiles(candidate):
                    return candidate

        # Если ничего не нашлось
        logging.warning("⚠️ SMILES не найден в ответе.")
        return None

    def process_dataframe(self, df, target_column="Target", output_column="Generated_SMILES"):
        generated_smiles = []

        for idx, row in df.iterrows():
            protein_sequence = row[target_column]
            prompt = self.generate_prompt(protein_sequence)
            logging.info(f"[{idx}] Prompt: {prompt[:200]}...")

            try:
                response = self.send_request(prompt)
                if response is None:
                    logging.warning(f"[{idx}] Ответ от модели пустой.")
                    generated_smiles.append(None)
                    continue

                content = response["choices"][0]["message"]["content"]
                logging.info(f"[{idx}] Raw Response: {content}")

                smiles = self.extract_smiles(content)
                if smiles and self.is_valid_smiles(smiles):
                    logging.info(f"[{idx}] ✅ Валидный SMILES: {smiles}")
                    generated_smiles.append(smiles)
                else:
                    logging.warning(f"[{idx}] ❌ Невалидный или отсутствующий SMILES.")
                    generated_smiles.append(None)

            except Exception as e:
                logging.error(f"[{idx}] 💥 Ошибка: {str(e)}")
                generated_smiles.append(None)

            time.sleep(1)

        df[output_column] = generated_smiles
        return df

    def save_results(self, df, output_file):
        df.to_csv(output_file, index=False)
        print(f"💾 Результаты сохранены в {output_file}")
        logging.info(f"Результаты сохранены в {output_file}")