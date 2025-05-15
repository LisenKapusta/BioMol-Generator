from pathlib import Path
import os

LOCAL_PATH = Path(__file__).parents[1]
DATASET_PATH = LOCAL_PATH / 'data/dataset.parquet'

TRAIN_DF_PATH = LOCAL_PATH / 'data' 'dataframes' / 'train_df.parquet'
VALID_DF_PATH = LOCAL_PATH / 'data' 'dataframes' / 'valid_df.parquet'
TEST_DF_PATH = LOCAL_PATH / 'data' 'dataframes' / 'test_df.parquet'

LOCAL_MODELS_PATH = LOCAL_PATH / 'models'
LOCAL_MODELS_PATH.mkdir(exist_ok=True, parents=True)

CHECKPOINTS_PATH = LOCAL_PATH / 'model_training/model_epoch_save'
CHECKPOINTS_PATH.mkdir(exist_ok=True, parents=True)

DOCUMENT_FILE_PATH = LOCAL_PATH / 'data' 'prompts' / 'document.txt'
SYSTEM = LOCAL_PATH / 'data' 'prompts' / 'system.txt'
PREPROMPT = LOCAL_PATH / 'data' / 'prompts' / 'preprompt.txt'
POSTPROMPT = LOCAL_PATH / 'data' 'prompts' / 'postprompt.txt'
TEMP = 0.1

LOCAL_SERVER_URL = "http://127.0.0.1:1234"
# LOCAL_MODEL = "qwen2.5-coder-14b-instruct-"
LOCAL_MODEL = 'qwen2.5-coder-14b-instruct'

LOCAL_MODEL_OUTPUT_FILES_PATH = LOCAL_PATH / 'data/local_model_results'

# API_KEY = os.getenv("API_KEY")
API_KEY = "sk-or-vv-296afe621cf9aaf6ef9f057aca893607d082e25d8b1ebd532664be7f67a59948"
API_MODEL_DEEPSEEK_CODER = "deepseek/deepseek-coder" 
DEEPSEEK_CODER_OUTPUT_FILE_PATH = LOCAL_PATH / "model_prompting/api_models/deepseek_coder_results.csv"
API_OUTPUT_FILES_PATH = LOCAL_PATH / 'data/api_models_results'

DEEPDTA_MODEL_PATH = LOCAL_PATH / 'DL4H/DeepDTA/source/combined_davis.h5'
LIGAND_PREDICTION_REPORT_PATH = LOCAL_PATH / 'check_results'

 # model = "mistralai/mistral-large-2411"
    # model = 'google/gemini-2.0-flash-001'