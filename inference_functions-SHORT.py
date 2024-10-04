import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm

def load_data():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/swim_lesson_data.csv")
    swim_lessons = np.array(df.iloc[:,0])
    drownings    = np.array(df.iloc[:,1])
    xy           = np.array(df.iloc[:,2])
    return swim_lessons,drownings,x,y

def load_more_data():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/swim_lesson_data.csv")
    swim_lessons        = np.array(df.iloc[:,0])
    drownings           = np.array(df.iloc[:,1])
    xy                  = np.array(df.iloc[:,2])
    distance_from_ocean = np.array(df.iloc[:,3])
    return swim_lessons,drownings,x,y,distance_from_ocean

def compute_residuals_2d(swim_lessons, drownings):    
    from statsmodels.formula.api import ols                    # import the required module
    dat                = {"x": swim_lessons, "y": drownings}   # define the predictor "x" and outcome "y"
    regression_results = ols("y ~ 1 + x", data=dat).fit()      # fit the model.
    residuals = regression_results.resid
    return residuals

def compute_residuals_3d(swim_lessons, drownings, distance_from_ocean):    
    from statsmodels.formula.api import ols                    # import the required module
    dat = {"w": distance_from_ocean, "x": swim_lessons, "y": drownings}
    regression_results = ols("y ~1 + w + x", data=dat).fit()
    residuals = regression_results.resid
    return residuals