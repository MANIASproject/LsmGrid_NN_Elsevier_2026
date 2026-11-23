import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from utils import ResultsManager
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ==================
#     FONTS!
# ==================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "pdf.fonttype": 42,   # TrueType en vez de Type 3
    "ps.fonttype": 42,
})

# ==================
#    LOAD DATA
# ==================
hcgp_manager = ResultsManager('results/lsmgrid_experiment_hcgp_aug10.json')


# ==================
#      FORMAT
# ==================
dataset_printable = {
    'FSDD':"FSDD",
    'NTIDIGITS':'N-TIDIGITS',
    'NMNIST':'N-MNIST',
    'MITBIH':'MIT-BIH'   
}

readout_printable = {
    'LOGREG':"LR",
    'RF': "RF",
    'SVM': "SVM",
    'KNN': "KNN"
}
# ==================
#      PLOT FN
# ==================
def plot_history(ax, dataset,readout):
    trials = [ trial['history'] for trial in hcgp_manager.results[f'{dataset}+{readout}'] ]
    for trial in trials:
        ax.plot( -np.array(trial) )


    y_label = 'Accuracy' if dataset!='MITBIH' else 'Sen≠N'

    ax.grid()
    ax.set_title(f'{dataset_printable[dataset]}+{readout_printable[readout]}')
    ax.set_ylabel(y_label)
    ax.set_xlabel('Iteration')
    ax.set_ylim(0.1,1)

fig,ax = plt.subplots(4,4,figsize=(25, 25))

# ==================
#   GENEREATE PLOT
# ==================
datasets = ['FSDD','NTIDIGITS','NMNIST','MITBIH']
readouts = ['LOGREG','RF','SVM','KNN']

for row,dataset in zip(ax,datasets):
    for col,readout in zip(row,readouts):
        plot_history(col, dataset,readout)

plt.savefig('analysis/outputs/convergence.pdf',  bbox_inches="tight")
