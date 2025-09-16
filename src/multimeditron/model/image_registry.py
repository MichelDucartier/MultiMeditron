from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os
import pyarrow.parquet as pq


class ImageParquetRegistry():
    def __init__(self, directory: str):
        parquet_files = self.find_parquet_files(directory)
        tables = [pq.read_table(file, columns=["path", "bytes", "format"]) for file in tqdm(parquet_files, desc="Loading parquet files")]
        combined = pq.concat_tables(tables)
        
        path_column = combined.column("path").to_pylist()  # Convert to Python list
        bytes_column = combined.column("bytes").to_pylist()
        format_column = combined.column("format").to_pylist()

        self.image_lookup = dict(zip(path_column, zip(bytes_column, format_column)))

    
    def find_parquet_files(self, directory):
        parquet_files = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".parquet"):
                parquet_files.append(os.path.join(directory, filename))

        return parquet_files


    def get_image(self, image_path):
        res = self.image_lookup.get(image_path, None)
        if res is None:
            return None
        
        img_as_bytes, img_format = res

        img = Image.open(BytesIO(img_as_bytes))

        return img

