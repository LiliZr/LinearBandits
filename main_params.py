import warnings
from itertools import product
from LinearBandits import *
from LinUCB import *
from LRP import *
from plots import *


#### PATH to results 
PATH = './results/LRP/parameters/'

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ######################## Params ###########################
    dataset = 'Random'
    # Number of loops we repeat experiments (To approximate in expectation)
    n_loop = 20
    seeds = np.arange(n_loop)


    ######################## Generate theta and action set ###########################
    ### Paramaters
    # Dimension
    d = 100
    # Max iteration
    n = 500
    # Number of arms
    m_actions = 2
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
    action_set = rng.randn(m_actions, d) 

    ######################## Models ###########################

    params_models = {   'LinUCB': ['scale', 'lam'],
                        'LRP': ['c0', 'lam', 'm', 'q', 'u', 'omega', 'C_lasso'],
                        'Random':[]
                        }

    scale_s = [(10**i) for i in range(-3, 3)]
    lam_s = [0.8]
    m_s = [30]
    q_s = [int(d*i) for i in [0.8]]
    C_lasso_s = [0.1]
    omega_s = [(10**i) for i in range(-3, 0)]

    params_values = {
                    'scale':scale_s,
                    'lam':lam_s,
                    'c0':[1],
                    'm':m_s,
                    'q':q_s,
                    'u':[2],
                    'omega':omega_s,
                    'C_lasso': C_lasso_s,
                }
    
    for model_ in [ LRP]:
        params_model = {}
        for p in params_models[model_.__name__ ]:
            params_model[p] = params_values[p]

        for vals in product(*params_model.values()):
            try:
                params = dict(zip(params_model, vals))
                params_str = ''.join([f'_{key}={value}' for key, value in params.items()])
                print(model_, params_str)
    
                models = [model_,  ]
                models_names = [m.__name__ for m in models]
                reward = {name: np.zeros((n_loop, n)) for name in models_names}
                regret = {name: np.zeros((n_loop, n)) for name in models_names}
                running_time = {name: np.zeros((n_loop, n)) for name in models_names}
                memory = {name: np.zeros((n_loop, n)) for name in models_names}


                for i in range(n_loop):
                    print(f'____It {i}___')
                    #### Optimal model
                    optimal = Optimal(theta, action_set=action_set, seed=seeds[i])
                    optimal.run(n)
                    reward_max = optimal.cumulative_reward
                    for model, model_name in zip(models, models_names):
                        model = model(theta, action_set=action_set, sigma=sigma, seed=seeds[i], delta=1/n, **params)
                        model.run(n)
                        regret[model_name][i] = reward_max - model.cumulative_reward
                        running_time[model_name][i] = model.time

                params_ = {
                    model_.__name__:params
                }
                ######################## Plots ########################
                plot_regret_time_over_iterations(models_names, regret, running_time, 
                                     n, d, m_actions, sigma, dataset, params_comparison=True, 
                                     PATH=PATH, params=params_)
            except Exception as exc:
                print(model_, params_str)
                print('\n__ERROR__:\n', exc)
                print('__________')
                pass