import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import plotly.graph_objects as go
from IPython.lib.display import YouTubeVideo
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript, Code

def load_data(using_colab=0):
    # Load the data.
    if using_colab:
        data        = sio.loadmat('/content/METER-Units/rodent_data.mat')
    else:
        data        = sio.loadmat('rodent_data.mat')       # Load the data default
    
    spikes  = data['spikes']                      # ... and define the variables.
    signals = data['signals']
    t       = data['t'][0];

    return spikes, signals, t