import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import math

def load_dataset(dir_name):
    df = pd.read_csv(dir_name, sep="\t")
    df.columns = ['a', 'b','label','tweet']
    #print("Shape of pandas frame: " + str(df.shape))
    return df['tweet'],df['label']

