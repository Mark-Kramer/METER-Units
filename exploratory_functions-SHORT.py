import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/refs/heads/main/rodent_data.csv")
    spikes  = df.iloc[:, 0:200].to_numpy()
    signals = df.iloc[:, 200:300].to_numpy()
    t       = df.iloc[:, 300].to_numpy()
    return spikes, signals, t