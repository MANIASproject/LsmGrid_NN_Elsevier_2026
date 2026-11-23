import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ResultsManager
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ================
#   LOAD DATA
# ================
managers = {
	"hcgp": ResultsManager('results/lsmgrid_experiment_hcgp_aug10.json'),
	'ga'  : ResultsManager('results/lsmgrid_experiment_ga_aug10.json'),
	'csa' :ResultsManager('results/lsmgrid_experiment_csa_aug10.json')
}

# ==================
#    NICE FONTS
# ==================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "pdf.fonttype": 42,   # TrueType en vez de Type 3
    "ps.fonttype": 42,
})
# ================
#  FORMAT PRINT
# ================
print_dataset = {
	'FSDD'		: 'FSDD',
	'NTIDIGITS'	: 'N-TIDIGITS',
	'NMNIST'	: 'N-MNIST',
	'MITBIH'	: 'MIT-BIH',
}

print_readout = {
	'LOGREG': 'LR',
	'RF'	: 'RF',
	'SVM'	: 'SVM',
	'KNN'	: 'KNN',
}

print_optimizer = {
	'hcgp': 'HC-GP',
	'ga'  : 'Parser+GA',
	'csa' : 'Parser+(A)CSA',
}

readouts   = ['LOGREG','RF','SVM','KNN']
datasets   = ['FSDD','NTIDIGITS','NMNIST','MITBIH']
optimizers = ['hcgp','ga','csa']

# ================
#    GET TRIALS
# ================
def get_trials(dataset, optimizer):
	trials  = []
	for readout in readouts:
		trials.append( [ (len(trial['best_grid']) ,trial['final_acc']) for trial in managers[optimizer].results[f'{dataset}+{readout}']] )
	return trials

def plot(trials):
	colors = ["#dea4f5","#ffe4b3",'#bceffb',"#335BAC"]
	for readout,trial,c in zip(readouts,trials,colors):
		neurons,accs = zip(*trial)
		plt.scatter(neurons,accs, label=print_readout[readout],color=c)

	plt.xlabel("Number of neurons")
	plt.ylabel('Accuracy')
	plt.grid()

# ================
#      PLOT
# ================
plt.figure(figsize=(20,16))

for row, optim in enumerate(optimizers):
	for col,dataset in enumerate(datasets):
		plt.subplot(3,4,4*row+col+1)

		if row==0:
			plt.annotate(print_dataset[dataset], xy=(0.5, 1), xycoords=plt.gca().title,
							xytext=(0, 15), textcoords="offset points",
							ha="center", va="bottom",
							fontsize=13, fontweight='bold')
			
		if col==0:
			plt.annotate(print_optimizer[optim], xy=(0, 0.5), xycoords=plt.gca().yaxis.label,
						xytext=(-15, 0), textcoords="offset points",
						ha="right", va="center", rotation=90,
						fontsize=13, fontweight='bold')

		trials = get_trials( dataset, optim )
		plot(trials)

		if row==0 and col==0:
			plt.legend()


plt.savefig('analysis/outputs/acc_neus.pdf',  bbox_inches="tight")
