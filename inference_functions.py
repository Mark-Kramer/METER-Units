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
        data        = sio.loadmat('/content/METER-Units/swim_lesson_data.mat')
    else:
        data         = sio.loadmat('swim_lesson_data.mat')       # Load the data default
    
    swim_lessons = data['swim_lessons']                      # ... and define the variables.
    drownings    = data['drownings']
    xy           = data['xy']
    distance_from_ocean = data['distance_from_ocean']

    return swim_lessons, drownings, xy

def load_more_data(using_colab=0):
    # Load the data.
    if using_colab:
        data        = sio.loadmat('/content/METER-Units/swim_lesson_data.mat')
    else:
        data         = sio.loadmat('swim_lesson_data.mat')       # Load the data default
    
    distance_from_ocean = data['distance_from_ocean']

    return distance_from_ocean

def plot_spatial_coordinates(xy, colors):
    import plotly.graph_objects as go

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

def create_dropdown_estimate_regression(swim_lessons, drownings):

    # Function to handle dropdown value change
    def on_dropdown_change(change):

        code = """
    from statsmodels.formula.api import ols                    # import the required module
    dat                = {"x": swim_lessons, "y": drownings}   # define the predictor "x" and outcome "y"
    regression_results = ols("y ~ 1 + x", data=dat).fit()      # fit the model.
    
    print('Slope estimate =',round(regression_results.params[1],3))     # Print the slope
    print('p-value        =',round(regression_results.pvalues[1],3))    # ... and the p-value.
    """
        
        with output:
            clear_output()
            if change['new'] == 'I want to write all the code myself.':
                print("\nGood for you! Here are some suggestions:\n")
                print("1. Consider using `ols` in `statsmodels`.")
                print("2. The outcome variable is `drownings`.")
                print("3. The predictor variable is `swim_lessons`.\n")
                print("If you get stuck, select the `Show me the code and the results.` option\n")
            elif change['new'] == 'Show me the code and I will run it myself.':
                print("\n Here's the code:\n")
                display(Code(data=code, language='python'))
            elif change['new'] == 'Show me the code and the results.':
                print("\n Here's the code:\n")
                display(Code(data=code, language='python'))
                print("\n And here are the results:\n")
                from statsmodels.formula.api import ols
                dat                = {"x": swim_lessons, "y": drownings}
                regression_results = ols("y ~ 1 + x", data=dat).fit()
                print('Slope estimate =',round(regression_results.params[1],5))
                print('p-value        =',round(regression_results.pvalues[1],10))
            elif change['new'] == 'Just show me the results!':
                print("\n Running code ... here are the results:\n")
                from statsmodels.formula.api import ols
                dat                = {"x": swim_lessons, "y": drownings}
                regression_results = ols("y ~ 1 + x", data=dat).fit()
                print('Slope estimate =',round(regression_results.params[1],5))
                print('p-value        =',round(regression_results.pvalues[1],10))
    
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

def create_dropdown_plot_regression(swim_lessons, drownings):

    # Function to handle dropdown value change
    def on_dropdown_change(change):
        code = """
    # Estimate the regression model.
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
    plt.figure(figsize=(12, 8))
    plt.scatter(swim_lessons,drownings)
    plt.plot(swim_lessons[indices_sorted[:,0]],mn[indices_sorted[:,0]], 'r')
    plt.plot(swim_lessons[indices_sorted[:,0]],ci_low[indices_sorted[:,0]], ':r')
    plt.plot(swim_lessons[indices_sorted[:,0]],ci_upp[indices_sorted[:,0]], ':r')
    plt.xlabel('Swim lessons')
    plt.ylabel('Drownings');
    """
        with output:
            clear_output()
            if change['new'] == 'I want to write all the code myself.':
                print("\nGood for you! Here are some suggestions:\n")
                print("1. Use the results of your regression estimate.")
                print("2. Plot the observed data, `swim_lessons` versus `drownings`")
                print("3. Plot your regression results (and 95% confidence intervals) on top of these data.\n")
                print("If you get stuck, select the `Show me the code and the results.` option\n")
            elif change['new'] == 'Show me the code and I will run it myself.':
                print("\n Here's the code:\n")
                display(Code(data=code, language='python'))
            elif change['new'] == 'Show me the code and the results.':
                print("\n Here's the code:\n")
                display(Code(data=code, language='python'))
                print("\n And here are the results:\n")
                plot_regression_results_2d(swim_lessons, drownings)
            elif change['new'] == 'Just show me the results!':
                print("\n Running code ... here are the results:\n")
                plot_regression_results_2d(swim_lessons, drownings)
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

def create_dropdown_estimate_regression_3(swim_lessons, drownings, distance_from_ocean):

    # Function to handle dropdown value change
    def on_dropdown_change(change):

        code = """
    from statsmodels.formula.api import ols                    # import the required module
    dat = {"w": distance_from_ocean, "x": swim_lessons, "y": drownings}
    regression_results_2_predictor = ols("y ~1 + w + x", data=dat).fit()
    
    print('Distance from ocean')
    print('Slope estimate =',round(regression_results_2_predictor.params[1],4))
    print('p-value        =',round(regression_results_2_predictor.pvalues[1],12))
    
    print('Number of swim lessons')
    print('Slope estimate =',round(regression_results_2_predictor.params[2],5))
    print('p-value        =',round(regression_results_2_predictor.pvalues[2],3))
    """
        
        with output:
            clear_output()
            if change['new'] == 'I want to write all the code myself.':
                print("\nGood for you! Here are some suggestions:\n")
                print("1. Consider using `ols` in `statsmodels`.")
                print("2. The outcome variable is `drownings`.")
                print("3. The predictor variables are `swim_lessons` and `distance_to_ocean`.\n")
                print("If you get stuck, select the `Show me the code and the results.` option\n")
            elif change['new'] == 'Show me the code and I will run it myself.':
                print("\n Here's the code:\n")
                display(Code(data=code, language='python'))
            elif change['new'] == 'Show me the code and the results.':
                print("\n Here's the code:\n")
                display(Code(data=code, language='python'))
                print("\n And here are the results:\n")
                from statsmodels.formula.api import ols                    # import the required module
                dat = {"w": distance_from_ocean, "x": swim_lessons, "y": drownings}
                regression_results_2_predictor = ols("y ~1 + w + x", data=dat).fit()
                print('Distance from ocean')
                print('Slope estimate =',round(regression_results_2_predictor.params[1],4))
                print('p-value        =',round(regression_results_2_predictor.pvalues[1],12))
                print('\nNumber of swim lessons')
                print('Slope estimate =',round(regression_results_2_predictor.params[2],5))
                print('p-value        =',round(regression_results_2_predictor.pvalues[2],3))
                                
            elif change['new'] == 'Just show me the results!':
                print("\n Running code ... here are the results:\n")
                from statsmodels.formula.api import ols                    # import the required module
                dat = {"w": distance_from_ocean, "x": swim_lessons, "y": drownings}
                regression_results_2_predictor = ols("y ~1 + w + x", data=dat).fit()
                print('Distance from ocean')
                print('Slope estimate =',round(regression_results_2_predictor.params[1],4))
                print('p-value        =',round(regression_results_2_predictor.pvalues[1],12))
                print('\nNumber of swim lessons')
                print('Slope estimate =',round(regression_results_2_predictor.params[2],5))
                print('p-value        =',round(regression_results_2_predictor.pvalues[2],3))
    
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


def plot_regression_results_2d(swim_lessons, drownings):
    
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
    plt.figure(figsize=(12, 8))
    plt.scatter(swim_lessons,drownings)
    plt.plot(swim_lessons[indices_sorted[:,0]],mn[indices_sorted[:,0]], 'r')
    plt.plot(swim_lessons[indices_sorted[:,0]],ci_low[indices_sorted[:,0]], ':r')
    plt.plot(swim_lessons[indices_sorted[:,0]],ci_upp[indices_sorted[:,0]], ':r')
    plt.xlabel('Swim lessons')
    plt.ylabel('Drownings')
    plt.show()

def plot_regression_results_3d(swim_lessons, drownings, distance_from_ocean):

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