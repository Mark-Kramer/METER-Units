import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def load_data():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/refs/heads/main/rodent_data.csv")
    spikes  = df.iloc[:, 0:200].to_numpy()
    signals = df.iloc[:, 200:300].to_numpy()
    t       = df.iloc[:, 300].to_numpy()
    return spikes, signals, t

def plot_spike_train(t, spikes):
    indices = [i for i, value in enumerate(spikes) if value == 1]; values  = [1] * len(indices)
    plt.plot(t[indices], values, 'ko');

def compute_p_values(spikes, signals):
    n1 = signals.shape[1]
    n2 = spikes.shape[1]
    p = np.zeros((n1, n2))

    for i in tqdm(range(n1)):       # tqdm shows the progress bar
        for j in range(n2):
                                    # GLM fitting with a Poisson family
            X = signals[:, i]       # Predictor variable
            y = spikes[:, j]        # Response variable
            X = sm.add_constant(X)  # Adding a constant column for the intercept
            glm_model = sm.GLM(y, X, family=sm.families.Poisson())
            glm_results = glm_model.fit()
            p[i, j] = glm_results.pvalues[1]  # Storing the p-value of the predictor

    return p