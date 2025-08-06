import pandas as pd

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/spindle_data_baseline.csv", header=None)
    baseline = df.to_numpy()
    baseline = baseline[0]

    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/spindle_data_during_treatment.csv", header=None)
    during = df.to_numpy()
    during = during[0]

    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/spindle_data_post_treatment.csv", header=None)
    post = df.to_numpy()
    post = post[0]
    
    return baseline, during, post