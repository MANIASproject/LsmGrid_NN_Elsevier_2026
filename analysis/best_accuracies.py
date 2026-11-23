import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from utils import ResultsManager
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ==============
#   LOAD DATA
# ==============
managers = {
	"hcgp": ResultsManager('results/lsmgrid_experiment_hcgp_aug10.json'),
	'ga'  : ResultsManager('results/lsmgrid_experiment_ga_aug10.json'),
	'csa' :ResultsManager('results/lsmgrid_experiment_csa_aug10.json')
}

# ==============
#   FUNCTIONS
# ==============
def count_synapses(grid):
    suma = 0
    for cell in grid:
        if not 'd3_lims' in cell:
            start = int(cell['d2_lims'][0])
            end  = int(cell['d2_lims'][1]) 
            suma += end-start
        else:
            start_d2 = int(cell['d2_lims'][0])
            end_d2   = int(cell['d2_lims'][1])
            start_d3 = int(cell['d3_lims'][0])
            end_d3   = int(cell['d3_lims'][1])
            suma += (end_d2-start_d2)*(end_d3-start_d3)
    return suma

def get_best_grid(target_dataset, only_lr):
    best_acc    = 0
    best_n_neu  = 0
    best_optim  = None
    best_config = None
    best_trial  = 0

    for optim_name in managers.keys():
        all_configs = managers[optim_name].results
        for config_name in all_configs:
            dataset,readout = config_name.split('+')
            if only_lr and readout!='LOGREG':
                continue
            if dataset!=target_dataset:
                continue
            for idx, trial in enumerate(all_configs[config_name]):
                trial_acc = np.round(trial['final_acc'],3)
                n_neu     = len(trial['best_grid']) 
                if (trial_acc>best_acc) or (trial_acc==best_acc and n_neu<best_n_neu):
                    best_acc    = trial_acc
                    best_n_neu  = n_neu
                    best_optim  = optim_name
                    best_config = config_name
                    best_trial  = idx

    return best_optim, best_config, best_trial
        
def print_stats(optim_name, config_name, best_trial, only_lr):
    best = managers[optim_name].results[config_name][best_trial]
    acc  = best['final_acc']
    neu  = len(best['best_grid'])
    syn  = count_synapses(best['best_grid'])

    title = 'LR best config    ' if only_lr else 'Global best config'
    print( f'({title}) Optim, readout, acc, #neu, $syn: ({optim_name:<4}, {config_name.split('+')[1]:<6}, {np.round(acc,3):<5}, {neu:<3}, {syn:<4})')

# ==============
#     PRINT
# ==============
for dataset_name in ['FSDD','NTIDIGITS','NMNIST','MITBIH']:
    print("="*10)
    print(dataset_name)
    print("="*10)
    optim_name, config_name, best_trial = get_best_grid(dataset_name, only_lr=False)
    print_stats(optim_name, config_name, best_trial, only_lr=False)
    optim_name, config_name, best_trial = get_best_grid(dataset_name, only_lr=True)
    print_stats(optim_name, config_name, best_trial, only_lr=True)

