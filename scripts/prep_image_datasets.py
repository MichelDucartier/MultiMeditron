from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfFileSystem
import json
import os
import pandas as pd
import zipfile

from typing import Dict, Union

# Specify here the hierarchy of datasets. The names have to fit folders of the MultiMediset repository.
dataset_folders = {
    "US": ["BUSI", "CAMUS", "COVID-US", "ct2", "DDTI", "US_SEG"],
    #"XR": ["chexpert", "iu_xray"]
}

# Specify here in which folder the datasets should be downloaded
path_datasets = os.path.abspath("../datasets/")

#path to reach the dataset within the repo.
#DO NOT put a / at the beginning or the end of the path.
#it is assumed to end with the name of the parent folder
path_to_dataset_repo = { 
    "BUSI": "image",
    "CAMUS": "image",
    "COVID-US": "image",
    "ct2": "image",
    "DDTI": "image",
    "US_SEG": "image",
    "chexpert": "image",
    "iu_xray": "image"
}

# Code to unzip a parquet file
failed = []
def write_parquet_to_folder(parquet_path: str, folder_path: str = "./") -> None:
    """
    Function maker that returns the function that writes images from bytes.

    Args:
      parquet_path: the path (absolute or relative) of the parquet file to read
      folder_path: the path (absolute or relative) of the folder that has to be created
    
    Returns:
      A function to write the file corresponding to a row object at folder_path.
      The "path" field of the row object is relative to folder_path.
    """

    folder_path = os.path.abspath(folder_path)
    
    def write_img(row: Dict[str, Union[str, bytes]]):
        """
        Write the file corresponding to a row.
        
        Args:
          row: A dictionary-like object with the following 3 fields:
               "path": path that the image should have, relative to folder_path
               "bytes": bytes of the file
               "format": format (used to deserialize the file)
        """

        path, img_as_bytes, img_format = row["path"], row["bytes"], row["format"]
        path = os.path.join(folder_path, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            img = Image.open(BytesIO(img_as_bytes))
            img.save(path, img_format)
        except: #unable to save the file for whatever reason…
            failed.append(path)

    table = pd.read_parquet(args.path_to_parquet, engine="pyarrow")
    table.apply(write_img(args.path_to_save), axis=1)

fs = HfFileSystem()
for folder, datasets in dataset_folders.items():
    for dataset in datasets:
        path_dataset = os.path.join(folder, dataset)
        os.makedirs(os.path.join(path_datasets, path_dataset), exist_ok=True)

        print(dataset)

        files = ["/".join(x["name"].split("/")[3:]) for x in fs.ls(f"hf://datasets/OpenMeditron/MultiMediset/{path_to_dataset_repo[dataset]}/{dataset}/")]
        
        #download the files from the hub
        for file in files:#[dataset+".jsonl", dataset+"_gpt.jsonl"] + zips:
            hf_hub_download(repo_id="OpenMeditron/MultiMediset", repo_type="dataset", filename=file, local_dir = folder)

        #extract the zips
        for zip_archive in filter(lambda x: x.endswith(".zip"), files):
            with zipfile.ZipFile(os.path.join(path_dataset, zip_archive), 'r') as zip_ref:
                zip_ref.extractall(path_dataset)
        
        #extract the parquets
        for parquet_archive in filter(lambda x: x.endswith(".parquet"), files):
            write_parquet_to_folder(os.path.join(path_dataset, parquet_archive), path_dataset)

def save_as_jsonl(dataset, path):
    with open(path, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

for folder, datasets in dataset_folders.items():
    for dataset in datasets:
        dataset_path = os.path.join(dataset, dataset.split("/")[-1])
        dataset_file = dataset_path+".jsonl"
        dataset_train = dataset_path+"-train.jsonl"
        dataset_test = dataset_path+"-test.jsonl"

        ds = load_dataset("json", data_files=dataset_file)["train"]
        split_ds = ds.train_test_split(train_size = train_rate, seed = 42)

        save_as_jsonl(split_ds["train"], dataset_train)
        save_as_jsonl(split_ds["test"], dataset_test)