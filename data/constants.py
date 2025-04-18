from pathlib import Path

LOCAL_PATH = Path(__file__).parents[1]
DATASET_PATH = LOCAL_PATH / 'data/dataset.parquet'

LOCAL_MODELS_PATH = LOCAL_PATH / 'models'
LOCAL_MODELS_PATH.mkdir(exist_ok=True, parents=True)

CHECKPOINTS_PATH = LOCAL_PATH / 'model_training/model_epoch_save'
CHECKPOINTS_PATH.mkdir(exist_ok=True, parents=True)

DOCUMENT_FILE_PATH = LOCAL_PATH / 'data/document.txt'
SYSTEM = LOCAL_PATH / 'data/system.txt'
PREPROMPT = LOCAL_PATH / 'data/preprompt.txt'
POSTPROMPT = LOCAL_PATH / 'data/postprompt.txt'
TEMP = 0.7

LOCAL_SERVER_URL = 'http://10.147.19.239:1234'