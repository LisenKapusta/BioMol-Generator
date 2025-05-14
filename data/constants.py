from pathlib import Path
import os

LOCAL_PATH = Path(__file__).parents[1]
DATASET_PATH = LOCAL_PATH / 'data/dataset.parquet'

TRAIN_DF_PATH = LOCAL_PATH / 'data/train_df.parquet'
VALID_DF_PATH = LOCAL_PATH / 'data/valid_df.parquet'
TEST_DF_PATH = LOCAL_PATH / 'data/test_df.parquet'

LOCAL_MODELS_PATH = LOCAL_PATH / 'models'
LOCAL_MODELS_PATH.mkdir(exist_ok=True, parents=True)

CHECKPOINTS_PATH = LOCAL_PATH / 'model_training/model_epoch_save'
CHECKPOINTS_PATH.mkdir(exist_ok=True, parents=True)

DOCUMENT_FILE_PATH = LOCAL_PATH / 'data/document.txt'
SYSTEM = LOCAL_PATH / 'data/system.txt'
PREPROMPT = LOCAL_PATH / 'data/preprompt.txt'
POSTPROMPT = LOCAL_PATH / 'data/postprompt.txt'
TEMP = 0.1

LOCAL_SERVER_URL = "http://127.0.0.1:1234"
# LOCAL_MODEL = "qwen2.5-coder-14b-instruct-"
LOCAL_MODEL = 'qwen2.5-coder-14b-instruct'

PROMPT_OUTPUT_FILE_PATH = LOCAL_PATH / 'model_prompting'
T5_V2_OUTPUT_FILE_PATH = LOCAL_PATH / 'model_training' / 'test' / 't5_v2_test.csv'
T5_V3_OUTPUT_FILE_PATH = LOCAL_PATH / 'model_training' / 'test' / 't5_v3_test.csv'

API_KEY = os.getenv("API_KEY")
API_MODEL_DEEPSEEK_CODER = "deepseek/deepseek-coder" 
DEEPSEEK_CODER_OUTPUT_FILE_PATH = LOCAL_PATH / "model_prompting/api_models/deepseek_coder_results.csv"

 # model = "mistralai/mistral-large-2411"
    # model = 'google/gemini-2.0-flash-001'