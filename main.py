import warnings

from LinearBandits import *
from LinUCB import *
from LRP import *
from plots import *


#### PATH to results 
PATH = './results/LRP/'

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    ######################## Params ###########################
    dataset = 'Random'
    # Number of loops we repeat experiments (To approximate in expectation)
    n_loop = 30
    seeds = np.arange(n_loop)


    ######################## Generate theta and action set ###########################
    ### Paramaters
    # Dimension
    d = 100
    # Max iteration
    n = 1000
    # Number of arms
    m = 2
    # Noise
    sigma = 0.3
    ### Generate Theta and action set
    # Theta 
    theta = np.zeros(d)
    values = [1, 2, 3, 4, 5, 1.1, 2.1, 3.1, 4.1, 5.1]
    for i, val in enumerate(values):
        theta[i] = val
    # Action set / Context vectors
    seed = 48
    rng = np.random.RandomState(seed)
    action_set = rng.randn(m, d) 

    ######################## Models ###########################



    #### Other models
    models = [Random, LinUCB, LRP]
    models_names = [m.__name__ for m in models]
    reward = {name: np.zeros((n_loop, n)) for name in models_names}
    regret = {name: np.zeros((n_loop, n)) for name in models_names}
    running_time = {name: np.zeros((n_loop, n)) for name in models_names}

    params = {'Random':
                {'lam':0},
            'LinUCB': 
                {'scale':0.01, 
                'lam':0.2},
            'LRP': 
                {'c0': 1,
                'lam': 0.8,
                'm':30,
                'q':80,
                'u':2,
                'omega': 0.01,
                'C_lasso': 0.1,
                }} 


    for i in range(n_loop):
        print(f'It {i}')
        #### Optimal model
        optimal = Optimal(theta, action_set=action_set)
        optimal.run(n)
        reward_max = optimal.cumulative_reward
        #### Other models
        for model, model_name in zip(models, models_names):
            model = model(theta, action_set=action_set, sigma=sigma, delta=1/n, seed=seeds[i], **params[model_name])
            model.run(n)
            regret[model_name][i] = reward_max - model.cumulative_reward
            running_time[model_name][i] = model.time


    ######################## Plots ########################
    print("## Saving plot ##")
    plot_regret_time_over_iterations(models_names, regret, running_time, 
                                     n, d, m, sigma, dataset, 
                                     PATH=PATH, params=params)
    
