# Load necessary packages
import pandas as pd

# Load the data
def load_data(N):                              # Load the data with sample size N
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/sample_size.csv")
    x = np.array(df.iloc[0:N-1,0])
    lifespan = np.array(df.iloc[0:N-1,1])
    return x,lifespan

def load_code():
    import requests
    url = "https://raw.githubusercontent.com/Mark-Kramer/METER-Units/main/sample_size_functions.py"
    response = requests.get(url)
    response.status_code
    code = response.text
    exec(code)

# Load the data in Google Colab
def load_data_Colab(N):                         # Load the data with sample size N
    data     = sio.loadmat('/content/METER-Units/sample_size.mat')   # Load the data
    x        = data['x']                        # ... and define the variables.
    lifespan = data['lifespan']
    
    x        = x[0:N]
    lifespan = lifespan[0:N]
    return x,lifespan