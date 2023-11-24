import math
import numpy as np
import os

from matplotlib import pyplot as plt

def define_label_title(model_name, params, 
                       d, n, m, dataset, params_comparison,
                       regret='', sigma=None):
    """
        return corresponding label and subtitle given params
    """
    label = f'{model_name}' 
    title = f'{regret}_{dataset}' if params_comparison else dataset
    title += f'_n={n}' 
    title += f'_d={d}_σ={sigma}_actions={m}' if dataset == 'Random' else ''
    sub_PATH = f'/d={d}_σ={sigma}_actions={m}_n={n}/' if dataset == 'Random' else ''
    sub_PATH += f'/{model_name}/' if params_comparison else ''


    if params_comparison:
        for key, value in params.items():
                title += f'_{key}={value}_' 
        title = title[:-1]

    return label, title, sub_PATH

    

def plot_regret_time_over_iterations(models_names, regret, running_time, 
                                     n, d, m, sigma, dataset, params_comparison=False, 
                                     PATH='./', params=None):

    
    sub_PATH = ''
    ######################## Plots ########################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    regret_final = 0

    ######## Plot models
    time_steps = np.arange(n)
    for model_name in models_names:
        ## Regret
        regret_mean = np.mean(regret[model_name], axis=0)
        regret_final = np.round(regret_mean[-1], decimals = 3)
        regret_std = np.std(regret[model_name], axis=0)
        # Define label and title
        label, title, sub_PATH = define_label_title(model_name, params[model_name], d, n, m, dataset, params_comparison, str(regret_final), sigma)
        ax1.plot(time_steps, regret_mean, label=label, marker='x', markersize=5, markevery=int(n/10))
        ax1.fill_between(time_steps, regret_mean - regret_std, regret_mean + regret_std, alpha=0.2)

        ## time
        time_mean = np.mean(running_time[model_name], axis=0)
        ax2.scatter(time_steps[-1], np.round(time_mean[-1], decimals = 3), label=np.round(time_mean[-1]), marker='x')
        ax2.plot(time_steps, time_mean, label=model_name)


       

    ax1.set_title(f'Cumulative Regret')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Cumulative Regret')
    ax1.legend()  

    ax2.set_title(f'CPU Time')
    ax2.set_xlabel('n')
    ax2.set_yscale('log')
    ax2.set_ylabel('Time (sec)')
    ax2.legend()  

    ##### Check path 
    PATH_image = f'{PATH}/{dataset}/{sub_PATH}/images/'

    if not os.path.exists(PATH_image):
        os.makedirs(PATH_image)

    # Image saving
    fig.suptitle(title, fontsize=14)
    fig.savefig(PATH_image + title + '.png', dpi=500)


    