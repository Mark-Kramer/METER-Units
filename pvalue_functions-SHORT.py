import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import plotly.graph_objects as go
from IPython.lib.display import YouTubeVideo
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript, Code
from tqdm import tqdm

def load_data():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/pvalue_baseline.csv")
    swim_lessons = np.array(df.iloc[:,0])
    drownings    = np.array(df.iloc[:,1])
    x            = np.array(df.iloc[:,3])
    y            = np.array(df.iloc[:,2])
    return swim_lessons,drownings,x,y

def load_data(using_colab=0):
    # Load the data.
    if using_colab:
        data        = sio.loadmat('/content/METER-Units/spindle_data.mat')
    else:
        data        = sio.loadmat('spindle_data.mat')       # Load the data default
    
    baseline          = data['baseline']
    during_treatment  = data['during_treatment']                      # ... and define the variables.
    post_treatment    = data['post_treatment']

    return baseline, during_treatment, post_treatment


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

import ipywidgets as widgets
from IPython.display import display, clear_output, Code
import scipy.stats as stats

def create_dropdown_compute_pvalues(during_treatment):
    # Dictionary to store results
    results = {'p_value': None}

    # Function to handle dropdown value change
    def on_dropdown_change(change):
        code = """
result = stats.ttest_1samp(during_treatment, popmean=0)
p_value_during = result.pvalue
"""
        
        with output:
            clear_output()
            if change['new'] == 'I want to write all the code myself.':
                print("\nGood for you! Here are some suggestions:\n")
                print("1. Consider using `ttest_1samp` in `statsmodels`.")
                print("2. The `popmean` is 0.")
                print("If you get stuck, select the `Show me the code and the results.` option\n")
            elif change['new'] == 'Show me the code and I will run it myself.':
                print("\nHere's the code:\n")
                display(Code(data=code, language='python'))
            elif change['new'] == 'Show me the code and the results.':
                print("\nHere's the code:\n")
                display(Code(data=code, language='python'))
                print("\nAnd here are the results:\n")
                result = stats.ttest_1samp(during_treatment, popmean=0)
                results['p_value'] = result.pvalue
                print("p-value: ", results['p_value'])
            elif change['new'] == 'Just show me the results!':
                print("\nRunning code ... here are the results:\n")
                result = stats.ttest_1samp(during_treatment, popmean=0)
                results['p_value'] = result.pvalue
                print("p-value: ", results['p_value'])

    # Create a dropdown widget
    dropdown = widgets.Dropdown(
        options=['I want to write all the code myself.', 'Show me the code and I will run it myself.', 'Show me the code and the results.', 'Just show me the results!'],
        value=None,
        description='Options:',
        disabled=False,
    )
    
    # Output widget to display text
    output = widgets.Output()
    
    # Register the event handler
    dropdown.observe(on_dropdown_change, names='value')
    
    # Display the dropdown widget and output widget
    display(dropdown)
    display(output)

    return results
