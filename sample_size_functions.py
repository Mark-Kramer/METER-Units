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

def do_nothing_function(N):
    if N <= 800:
        print("You stare at the jumbled results, numbers and graphs swirling into a chaotic mess.")
        print("")
        print("The data makes no sense, the patterns are meaningless.")
        print("")
        print("You realize with a sinking feeling that your choice of sample size was too small, too narrow to capture the true picture.")
        print("")
        print("The project, once so full of promise, is now a monument to your miscalculation.")
        print("")
        print("You try to backtrack, to salvage something from the wreckage, but it's too late.")
        print("")
        print("The deadline looms, and there is no time to start over.")
        print("")
        print("The project ends here, in a tangle of flawed data and lost opportunities.")
        print("")
        print("If only you had chosen a larger sample size, maybe things would have been different.")
        print("")
        print("But now, all you can do is close the book and learn from your mistake.")
        print("")
        print("THE END.")
        print("")

    elif 800 < N <= 1200:
        print("Congratulations! You've reached the final page of your statistical adventure.")
        print("")
        print("Your choice of sample size led to successful results.")
        print("")
        print("The data aligns with the hypothesized result, the analysis checks out, and everything seems to be in order.")
        print("")
        print("You might feel a sense of triumph and relief wash over you.")
        print("")
        print("But there's an uneasy feeling lingering in the back of your mind.")
        print("")
        print("How did you arrive at the correct sample size?")
        print("")
        print("Was it a calculated decision based on a thorough understanding of the principles, or did luck play a significant role?")
        print("")
        print("As you stand on the brink of your next project, remember this moment.")
        print("")
        print("This time, you might have gotten lucky. But next time, the stakes might be higher, and luck might not be on your side.")
        print("")
        print("Take this as a lesson: delve deeper into the principles of sample size determination.")
        print("")
        print("Equip yourself with the knowledge to make informed decisions. Your future success depends on it.")
        print("")
        print("The adventure continues, and the next chapter is yours to write.")
        print("")
        print("Will you rely on luck again, or will you master the art of sample size calculations? The choice is yours.")
        print("")
        print("THE END.")
        print("")

    else:
        print("Congratulations! You've reached the final page of your statistical adventure.")
        print("")
        print("Your choice of sample size has yielded accurate results.")
        print("")
        print("The data aligns with the hypothesized result, the analysis checks out, and everything seems to be in order.")
        print("")
        print("However, as you review the costs, the reality of your decision sets in.")
        print("")
        print("The sample size you chose was much larger than necessary.")
        print("")
        print("The data collection process, while thorough, has left you with an extraordinary bill.")
        print("")
        print("The expenses have skyrocketed beyond your initial budget, and now, your lab faces the consequences.")
        print("")
        print("To pay for the costly experiment, significant cutbacks are needed.")
        print("")
        print("Resources are reallocated, projects are postponed, and the lab's progress is stunted.")
        print("")
        print("The success of your experiment is overshadowed by the financial strain it has caused.")
        print("")
        print("As you reflect on this experience, remember the importance of balance.")
        print("")
        print("A successful experiment is not just about accurate results; it's also about efficient resource management.")
        print("")
        print("Learn from this costly lesson and strive to find the optimal sample size that balances accuracy with affordability.")
        print("")
        print("The adventure continues, and the next chapter is yours to write.")
        print("")
        print("Will you learn to balance precision with practicality, or will you let costs spiral out of control again?")
        print("")
        print("The choice is yours.")
        print("")
        print("THE END.")
        print("")






    