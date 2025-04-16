from pathlib import Path

LOCAL_PATH = Path(__file__).parents[1]
DATASET_PATH = LOCAL_PATH / 'data/dataset.parquet'

LOCAL_MODELS_PATH = LOCAL_PATH / 'models'
LOCAL_MODELS_PATH.mkdir(exist_ok=True, parents=True)

CHECKPOINTS_PATH = LOCAL_PATH / 'model_training/model_epoch_save'
CHECKPOINTS_PATH.mkdir(exist_ok=True, parents=True)
