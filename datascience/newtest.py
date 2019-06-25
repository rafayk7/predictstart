import pandas as pd
import numpy as np

data = pd.read_csv('ks-projects-201612.csv', encoding='cp1252')
def load_and_explore(data):
    print("############# PREVIEW ########################")
    print(data.head())
    print("############# DATA TYPES ########################")
    print(data.info())
    print("############# NO. OF UNIQUE VALS ########################")
    print(data.nunique())

# load_and_explore(data)
