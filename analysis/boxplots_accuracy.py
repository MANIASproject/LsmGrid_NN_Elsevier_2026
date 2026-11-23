import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ResultsManager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

# ==============
#   LOAD DATA
# ==============
managers = {
	"hcgp": ResultsManager('results/lsmgrid_experiment_hcgp_aug10.json'),
	'ga'  : ResultsManager('results/lsmgrid_experiment_ga_aug10.json'),
	'csa' :ResultsManager('results/lsmgrid_experiment_csa_aug10.json')
}

def get_accs(dataset,readout,algo):
	return [ trial['final_acc'] for trial in managers[algo].results[f'{dataset}+{readout}'] ]

# ==================
#    NICE FONTS
# ==================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "pdf.fonttype": 42, 
    "ps.fonttype": 42,
})
# ==============================================================
#   COMPARISON BY DATASET, DIFFERENT READOUTS AND OPTIMIZERS
# ==============================================================

def plot_bloxplot(dataset, with_legend):
    readouts = ['LOGREG','RF','SVM','KNN']
    algos 	 = ['hcgp','ga','csa']

    all_accs = []
    for r in readouts:
        for a in algos:
            all_accs.append( get_accs(dataset,r,a) )
        all_accs.append([])

    bp = plt.boxplot(all_accs, patch_artist=True)


    colors = ["#d91111","#f7e943","#093da4",'white']
    for idx,box in enumerate(bp['boxes']):
        box.set_facecolor( colors[idx%len(colors)] )

    if with_legend:
        plt.legend(
            handles=[
                Line2D([0],[0], marker='o',color='w',markerfacecolor=colors[0], markersize=10, lw='1', label='HC-GP'),
                Line2D([0],[0], marker='o',color='w',markerfacecolor=colors[1], markersize=10, lw='1', label='Parser+GA'),
                Line2D([0],[0], marker='o',color='w',markerfacecolor=colors[2], markersize=10, lw='1', label='Parser+(A)CSA'),
            ],title='Optimizer')

    plt.xticks([4*i+2 for i in range(0,4)], labels=['LR','RF','SVM','KNN'])
    plt.grid()
    plt.ylabel('Accuracy')
    plt.xlabel('Readout layer')

    names    = {'FSDD':'FSDD','NTIDIGITS':'N-TIDIGITS','NMNIST':'N-MNIST','MITBIH':'MIT-BIH'}
    plt.title(names[dataset])


# ================
#   BUILD PLOTS
# ================

datasets = ['FSDD','NTIDIGITS','NMNIST','MITBIH']

plt.figure(figsize=(20,4))

for idx,dataset in enumerate(datasets):
    plt.subplot(1,4,idx+1)
    plot_bloxplot(dataset, with_legend=idx==0)

plt.savefig('analysis/outputs/boxplot_accuracies.pdf',  bbox_inches="tight")