import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm

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

def fdr(p):
    # List of p-values
    p_values = p.flatten()
    
    # Desired false discovery rate level
    q = 0.05
    
    # Sort p-values and get the sorted indices
    sorted_indices  = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Number of multiple tests
    m = len(p_values)
    
    # Calculate the Benjamini-Hochberg critical values
    critical_values = (np.arange(1, m+1) / m) * q
    
    # Find the largest p-value that is smaller than the critical value
    max_significant = np.max(np.where(sorted_p_values <= critical_values))
    
    # Initialize a boolean array for significance
    is_significant = np.zeros(m, dtype=bool)

    # If any p-values are significant, set them as True
    if max_significant >= 0:
        is_significant[sorted_indices[:max_significant+1]] = True
    
    # Make matrix to indicate significant p-value after FDR.
    p_values_signficant_after_FDR = np.zeros(np.shape(p))
    np.shape(p_values_signficant_after_FDR)
    p_values_signficant_after_FDR[np.unravel_index(np.where(is_significant==True), np.shape(p))] = 1

    return p_values_signficant_after_FDR