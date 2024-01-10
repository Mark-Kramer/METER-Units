# Load necessary packages
import scipy.io as sio

# Load the data
def load_data(N):                         # Load the data with sample size N
    data     = sio.loadmat('sample_size.mat')  # Load the data
    x        = data['x']                       # ... and define the variables.
    lifespan = data['lifespan']
    
    x        = x[0:N]
    lifespan = lifespan[0:N]
    return x,lifespan