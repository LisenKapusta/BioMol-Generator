from pathlib import Path
from datasets import load_dataset
from constants import DATASET_PATH

ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered")
parquet_filename = DATASET_PATH
# Преобразование каждого раздела в Parquet
for split, dataset in ds.items():
    dataset.to_parquet(parquet_filename)
    print(f"Saved {parquet_filename}")
    