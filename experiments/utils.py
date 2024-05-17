import torch
from torch import nn
import numpy as np
import os
import pandas as pd

def save_to_csv(df: pd.DataFrame, file_path: str):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if not os.path.isfile(file_path):
        # If the file doesn't exist, write the DataFrame with headers
        df.to_csv(file_path, index=False)
    else:
        # If the file exists, append the DataFrame without headers
        df.to_csv(file_path, mode='a', header=False, index=False)

