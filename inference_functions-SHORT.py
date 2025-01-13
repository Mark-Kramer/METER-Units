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
    x            = np.array(df.iloc[:,2])
    y            = np.array(df.iloc[:,3])
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

def plot_spatial_coordinates(xy, colors):
    #import plotly.graph_objects as go

    # Example x-y coordinates
    x_coordinates = xy[:,1]
    y_coordinates = xy[:,0]
    # Create a scattermapbox trace
    trace = go.Scattermapbox(
        lat=y_coordinates,
        lon=x_coordinates,
        mode='markers',
        marker=dict(
            size=10,
            color=colors, #residuals.to_numpy(),
            colorscale='RdYlBu',  # Choose a colorscale (Red-Blue in this case)
            cmin=-0.25, #min(residuals.to_numpy()),
            cmax= 0.25, #max(residuals.to_numpy()),
            colorbar=dict(title='Variable'),
            opacity=0.6
        ),
    )

    # Define the layout for the map
    layout = go.Layout(
        mapbox=dict(
            center=dict(lat=sum(y_coordinates)/len(y_coordinates), lon=sum(x_coordinates)/len(x_coordinates)),
            zoom=9,
            style='open-street-map'  # You can change the map style
        ),
        title='X-Y Coordinates on Map'
    )
    
    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(width=800, height=600)
    
    # Show the plot
    fig.show();