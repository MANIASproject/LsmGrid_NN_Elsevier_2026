import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from utils import ResultsManager
from scipy.stats import mannwhitneyu
from matplotlib import font_manager

# =============================
#         NICE FONTS
# =============================

font_manager.fontManager.addfont("fonts/Nunito-Regular.ttf")
font_manager.fontManager.addfont("fonts/Nunito-Bold.ttf")
plt.rcParams["font.family"] = "Nunito"
plt.rcParams["pdf.fonttype"] = 42

# =============================
#   FUNCTIONS TO PROCESS DATA
# =============================
def get_pvalues_matrix(dataset):
    manager_hcgp = ResultsManager('results/lsmgrid_experiment_hcgp_aug10.json')
    manager_ga   = ResultsManager('results/lsmgrid_experiment_ga_aug10.json')
    manager_csa  = ResultsManager('results/lsmgrid_experiment_csa_aug10.json')
    
    ml_algos = ['LOGREG','RF','SVM','KNN']
    optims   = [manager_hcgp, manager_ga, manager_csa]

    all_accs = []
    for algo in ml_algos:
        for manager in optims:
            accuracies = [ trial['final_acc'] for trial in manager.results[f'{dataset}+{algo}'] ]
            all_accs.append( accuracies )


    p_values = np.zeros((12,12))
    for i,row in enumerate(all_accs):
        for j,col in enumerate(all_accs):
            _, p = mannwhitneyu(row, col, alternative="greater")
            p_values[i,j] = p
    p_values = p_values.round(3)

    return p_values

def intra_readout_best(data,offset):
    all_best = []
    i = 3*offset
    j = 3*offset
    if data[i,j+1]<0.05 and data[i,j+2]<0.05:
        all_best.append((j,i))
        all_best.append((j+1,i))
        all_best.append((j+2,i))

    if data[i+1,j]<0.05 and data[i+1,j+2]<0.05:
        all_best.append((j,i+1))
        all_best.append((j+1,i+1))
        all_best.append((j+2,i+1))

    if data[i+2,j]<0.05 and data[i+2,j+1]<0.05:
        all_best.append((j,i+2))
        all_best.append((j+1,i+2))
        all_best.append((j+2,i+2))

    return all_best

def intra_readout_worst(data,offset):
    all_worst = []
    i = 3*offset
    j = 3*offset

    if data[i+1,j]<0.05 and data[i+2,j]<0.05:
        all_worst.append((j,i))
        all_worst.append((j,i+1))
        all_worst.append((j,i+2))

    if data[i,j+1]<0.05 and data[i+2,j+1]<0.05:
        all_worst.append((j+1,i))
        all_worst.append((j+1,i+1))
        all_worst.append((j+1,i+2))

    if data[i,j+2]<0.05 and data[i+1,j+2]<0.05:
        all_worst.append((j+2,i))
        all_worst.append((j+2,i+1))
        all_worst.append((j+2,i+2))

    return all_worst

def inter_readout_best(data, i,j):
    i = 3*i
    j = 3*j
    all_best = []
    if data[i,j]<0.05 and data[i+1,j+1]<0.05 and data[i+2,j+2]<0.05:
        all_best.append( (j,i) )
        all_best.append( (j+1,i+1) )
        all_best.append( (j+2,i+2) )
    return all_best

# ============================
#   FUNCTION TO BUID TABLE 
# ============================

def comp_table(p_values,ax):
    table = np.full((12, 12, 3), 1.0)
    ax.imshow(table)

    # =====================
    #  INTRA READOUT WORST
    # =====================
    intra_readout = []
    for i in range(4):
        intra_readout += intra_readout_worst(p_values,i)

    for cell in np.array(intra_readout):
        ax.add_patch(FancyBboxPatch(cell - 0.5, width=1, height=1,
                    boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#e7c2f6", edgecolor="none", alpha=0.5))


    # =====================
    #  INTRA READOUT BEST
    # =====================
    intra_readout = []
    for i in range(4):
        intra_readout += intra_readout_best(p_values,i)

    for cell in np.array(intra_readout):
        ax.add_patch(FancyBboxPatch(cell - 0.5, width=1, height=1,
                    boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#fcf3e2", edgecolor="none", alpha=0.8))

    # ==================
    #   INTER READOUT   
    # ==================
    inter_readout = []
    for i in range(4):
        for j in range(4):
            if i!=j:
                inter_readout += inter_readout_best(p_values,i,j)

    for cell in np.array(inter_readout):
        ax.add_patch(ax.add_patch(FancyBboxPatch(cell - 0.5, width=1, height=1,
                                    boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor="#bceffb", edgecolor="none", alpha=1)))

    # ==================
    #      CELLS
    # ==================
    for i in range(12):
        for j in range(12):
            ax.text(j, i, p_values[i, j], ha="center", va="center", fontfamily='Nunito')

    for i in range(4):
        for j in range(4):
            ax.add_patch(  Rectangle((i*3-0.5,j*3-0.5), width=3, height=3, fill=False, linewidth=0.5) )

    for i in range(4):
        ax.add_patch(  Rectangle((i*3-0.5,i*3-0.5), width=3, height=3, fill=False, linewidth=2) )

    # ================
    # AXIS AND LABELS
    # ================
    # FIRST LEVEL
    labels = ["hcgp", "ga", "csa"] * 4
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticks(range(12))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(12))
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()


    # SECOND LEVEL
    centers = [1, 4, 7, 10]
    labels = ["LR", "RF", "SVM","KNN"]
    # x axis
    sec_x = ax.secondary_xaxis(1.05)
    sec_x.set_xticks(centers)
    sec_x.set_xticklabels(labels, fontweight="bold")
    sec_x.tick_params(length=0)
    sec_x.spines["top"].set_visible(False)

    # y axis
    sec_y = ax.secondary_yaxis(-0.05)
    sec_y.set_yticks(centers)
    sec_y.set_yticklabels(labels, fontweight="bold")
    sec_y.tick_params(length=0)
    sec_y.spines["left"].set_visible(False)

# ============================
#       BUILDS TABLES
# ============================
fig, axes = plt.subplots(2,2,figsize=(16,16))

all_datasets = ['FSDD','NTIDIGITS','NMNIST','MITBIH']

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

idx = 0
for row in range(2):
    for col in range(2):
        d_name = all_datasets[idx]
        axes[row,col].set_title( d_name, fontsize=16, fontweight="bold", fontfamily='Nunito' )
        p_values = get_pvalues_matrix( d_name )
        comp_table(p_values, axes[row,col])
        idx += 1

plt.savefig('analysis/outputs/pvalues.pdf', bbox_inches="tight")    