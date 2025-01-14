import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols

def load_data():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/swim_lesson_data.csv")
    swim_lessons = np.array(df.iloc[:,0])
    drownings    = np.array(df.iloc[:,1])
    x            = np.array(df.iloc[:,3])
    y            = np.array(df.iloc[:,2])
    return swim_lessons,drownings,x,y

def load_more_data():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/Mark-Kramer/METER-Units/master/swim_lesson_data.csv")
    swim_lessons        = np.array(df.iloc[:,0])
    drownings           = np.array(df.iloc[:,1])
    x                   = np.array(df.iloc[:,3])
    y                   = np.array(df.iloc[:,2])
    distance_from_ocean = np.array(df.iloc[:,4])
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

def plot_spatial_coordinates(x, y, colors):
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Define the projection for the map
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.set_extent([min(x) - 0.25, max(x) + 0.25, min(y) - 0.25, max(y) + 0.25], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')
    
    # Plot the points
    sc = plt.scatter(x, y, c=colors, cmap='RdYlBu', s=40, vmin=-0.25, vmax=0.25, alpha=0.6, edgecolor='k', linewidth=0.5, transform=ccrs.PlateCarree())
    
    # Add a colorbar
    cbar = plt.colorbar(sc, orientation='vertical', pad=0.05)
    cbar.set_label('Variable')
    
    # Add a title
    plt.title('X-Y Coordinates on Map', fontsize=16)
    
    # Show the plot
    plt.show()

def estimate_line(swim_lessons, drownings):
    dat                = {"x": swim_lessons, "y": drownings}   # define the predictor "x" and outcome "y"
    regression_results = ols("y ~ 1 + x", data=dat).fit()      # fit the model.
    m                = regression_results.params[1]
    m_standard_error = regression_results.bse[1]
    return m, m_standard_error

def plot_line(swim_lessons, drownings):
    
    from statsmodels.formula.api import ols                    # import the required module
    dat                = {"x": swim_lessons, "y": drownings}   # define the predictor "x" and outcome "y"
    regression_results = ols("y ~ 1 + x", data=dat).fit()      # fit the model.
    
    # Get model prediction.
    pred   = regression_results.get_prediction().summary_frame()
    mn     = pred['mean']
    ci_low = pred['mean_ci_lower'] 
    ci_upp = pred['mean_ci_upper']
    
    # And plot it.
    indices_sorted = np.argsort(swim_lessons,0)
    plt.figure(figsize=(6, 4))
    plt.scatter(swim_lessons,drownings)
    plt.plot(swim_lessons[indices_sorted],mn[indices_sorted], 'r')
    plt.plot(swim_lessons[indices_sorted],ci_low[indices_sorted], ':r')
    plt.plot(swim_lessons[indices_sorted],ci_upp[indices_sorted], ':r')
    plt.xlabel('Swim lessons')
    plt.ylabel('Drownings')
    plt.show()

def plot_line_with_residuals(swim_lessons, drownings):
    
    from statsmodels.formula.api import ols                    # import the required module
    dat                = {"x": swim_lessons, "y": drownings}   # define the predictor "x" and outcome "y"
    regression_results = ols("y ~ 1 + x", data=dat).fit()      # fit the model.
    
    # Get model prediction.
    pred   = regression_results.get_prediction().summary_frame()
    mn     = pred['mean']
    
    # And plot it.
    indices_sorted = np.argsort(swim_lessons,0)
    plt.figure(figsize=(6, 4))
    plt.scatter(swim_lessons,drownings)
    plt.plot(swim_lessons[indices_sorted],mn[indices_sorted], 'r')
    plt.xlabel('Swim lessons')
    plt.ylabel('Drownings')
    
    # Draw orange lines from each data point to the regression line
    for i in range(len(swim_lessons)):
        plt.plot([swim_lessons[i], swim_lessons[i]],  # x-coordinates
                 [drownings[i], mn[i]],              # y-coordinates
                 color='orange', linestyle='--', linewidth=0.8, zorder=0)
    plt.show()

def estimate_plane(swim_lessons, drownings, distance_from_ocean):
    dat = {"w": distance_from_ocean, "x": swim_lessons, "y": drownings}
    regression_results= ols("y ~1 + x + w", data=dat).fit()
    m1                = regression_results.params[1]
    m1_standard_error = regression_results.bse[1]
    m2                = regression_results.params[2]
    m2_standard_error = regression_results.bse[2]
    return m1, m1_standard_error, m2, m2_standard_error

def plot_plane(swim_lessons, drownings, distance_from_ocean):

    # Create a meshgrid for 3D plotting
    x1 = np.transpose(distance_from_ocean)[0];
    x2 = np.transpose(swim_lessons)[0];
    y  = np.transpose(drownings)[0]; 
    x1_range = np.linspace(x1.min(), x1.max(), 100)
    x2_range = np.linspace(x2.min(), x2.max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    
    # Predict the response for each point in the meshgrid
    dat = {"w": distance_from_ocean, "x": swim_lessons, "y": drownings}
    from statsmodels.formula.api import ols                    # import the required module
    regression_results_2_predictor = ols("y ~1 + w + x", data=dat).fit()
    coefficients = regression_results_2_predictor.params
    y_pred_mesh = coefficients[0] + coefficients[1] * x1_mesh + coefficients[2] * x2_mesh
    
    # Create an interactive 3D plot using plotly
    fig = go.Figure()
    
    # Scatter plot for data points
    fig.add_trace(go.Scatter3d(
        x=x1,
        y=x2,
        z=y,
        mode='markers',
        marker=dict(size=5, color='red') #,
    #    name='Data Points'
    ))
    
    # Surface plot for OLS regression surface
    fig.add_trace(go.Surface(
        x=x1_mesh,
        y=x2_mesh,
        z=y_pred_mesh,
        #colorscale='blues',
        opacity=0.7,
        name='OLS Surface'
    ))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Distance from ocean',
            yaxis_title='Swim lessons',
            zaxis_title='Drownings',
        )
    )
    
    fig.update_layout(width=800, height=600)
    
    # Show the interactive plot
    fig.show()
    
    # Show the plot
    plt.show()