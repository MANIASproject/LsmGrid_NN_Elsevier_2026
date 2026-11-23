import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ResultsManager, plot_grid2D, plot_grid3D
from grids import Cell
import matplotlib.pyplot as plt

# ============================
#         FONTS!
# ============================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "pdf.fonttype": 42,   # TrueType en vez de Type 3
    "ps.fonttype": 42,
})
# ============================
#  DEFINITIONS AND FUNCTIONS
# ============================
all_managers = {
    'hcgp' : ResultsManager('results/lsmgrid_experiment_hcgp_aug10.json'),
    'ga'   : ResultsManager('results/lsmgrid_experiment_ga_aug10.json'),
    'csa'   : ResultsManager('results/lsmgrid_experiment_csa_aug10.json'),
}

def get_best_grid(configuration):
	manager,dataset,readout = configuration
	all_config = all_managers[manager].results
	trials = [ (trial['final_acc'], trial['best_grid']) for trial in all_config[f'{dataset}+{readout}'] ]
	best   = sorted(trials, key=lambda item: (item[0], -len(item[1])))[-1]

	print(f"Accuracy {best[0]}    neurons: {len(best[1])}")
	g_json  = best[1]

	g = []
	for c in g_json:
		if  'd3_lims' in c:
			g.append( Cell(t_lims=c['t_lims'], d2_lims=c['d2_lims'], d3_lims=c['d3_lims']) )
		else:
			g.append( Cell(t_lims=c['t_lims'], d2_lims=c['d2_lims']) )
	return g



# ============================
#      GET GRIDS TO PLOT
# ============================
configutations_to_eval = [
    ('hcgp','FSDD'     ,'KNN'),
    ('ga'  ,'NTIDIGITS','LOGREG'),
    ('ga'  ,'NMNIST','LOGREG'),
    ('ga'  ,'MITBIH','RF'),
]

g_fsdd      = get_best_grid(configutations_to_eval[0])
g_ntidigits = get_best_grid(configutations_to_eval[1])
g_nmnist    = get_best_grid(configutations_to_eval[2])
g_mitbih    = get_best_grid(configutations_to_eval[3])

# ============================
#     PLOT PLOT PLOT!
# ============================
plt.figure(figsize=(20,12))

def plot_grid(g, pos, title=None, xlabel="Time", ylabel=None, time=None, mit_bih=False):
    plt.subplot(pos)
    if time is not None:    plot_grid3D(g, time, color='black')
    else:                   plot_grid2D(g, color='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if mit_bih:
        plt.yticks([0,1], labels=["-","+"])

plot_grid(g_fsdd      , 231, title="FSDD", ylabel="Dimension 2 (Audio channel)")
plot_grid(g_ntidigits , 232, title="N-TIDIGITS", ylabel="Dimension 2 (Audio channel)")
plot_grid(g_mitbih    , 233, title="MIT-BIH", ylabel="Dimension 2 (Fixed ; Delta encoding channels)", mit_bih=True)
plot_grid(g_nmnist    , 234, title="N-MNIST (t=0ms)"  ,xlabel="Dimension 2 (width)",ylabel="Dimension 3 (height)",time=0)
plot_grid(g_nmnist    , 235, title="N-MNIST (t=175ms)",xlabel="Dimension 2 (width)",ylabel="Dimension 3 (height)",time=175)
plot_grid(g_nmnist    , 236, title="N-MNIST (t=349ms)",xlabel="Dimension 2 (width)",ylabel="Dimension 3 (height)",time=349)

plt.savefig('analysis/outputs/best_grids.pdf',  bbox_inches="tight")
