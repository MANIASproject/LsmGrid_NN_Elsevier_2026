
import sys
sys.path.append("../..")

from grids import generate_network, convert_coords_to_2d
from optimizer import parse_grid, CSA
from utils import evaluate_kfold

import sim_ballena as ballena
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import pickle
import datetime


# ===============
#  IRACE CONFIG
# ===============
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

# Get the parameters as command line arguments.
configuration_id = sys.argv[1]
instance_id      = sys.argv[2]
seed             = sys.argv[3]
instance         = sys.argv[4]
cand_params      = sys.argv[5:]


# Parse parameters
while cand_params:
    # Get and remove first and second elements.
    param = cand_params.pop(0)
    value = cand_params.pop(0)
    if param == "--scale":
        scale = float(value)
    elif param == "--ap":
        ap = float(value)
    elif param == "--n_mut":
        n_mut = int(value)
    elif param == "--max_grace":
        max_grace = int(value)
    else:
        target_runner_error("unknown parameter %s" % (param))


# ==============
#  LOAD DATASET
# ==============

def NMNIST_2D(instance):
    return [ (t, convert_coords_to_2d(x,y,34)) for t,(x,y) in instance]

with open(instance,'r') as file:
    dataset_name,readout_name = file.read().strip().split('+')

with open(f'../datasets/{dataset_name}.pickle','rb') as file:
    dataset = pickle.load(file)

if dataset_name=='nmnist':
    dataset['d'] = [ ballena.Instance(NMNIST_2D(x)) for x in dataset['d'] ]
else:
    dataset['d'] = [ ballena.Instance(x) for x in dataset['d'] ]

if readout_name=='logreg':	readout = LogisticRegression(max_iter=500, solver='saga')
elif readout_name=='rf'	 :	readout = RandomForestClassifier(n_estimators=30)
elif readout_name=='svm' :	readout = SVC( kernel='poly', degree=2 )
elif readout_name=='knn' : 	readout = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')

best_metric = {
    "fsdd+logreg": 0.9417,
    "fsdd+rf": 0.9617,
    "fsdd+knn": 0.975,
    "fsdd+svm": 0.98,
    "ntidigits+logreg": 0.8258,
    "ntidigits+rf": 0.8001,
    "ntidigits+svm": 0.8126,
    "ntidigits+knn": 0.7647,
    "nmnist+logreg": 0.954,
    "nmnist+rf": 0.938,
    "nmnist+svm": 0.943,
    "nmnist+knn": 0.945,
    "mitbih+logreg": 0.8923,
    "mitbih+rf": 0.9711,
    "mitbih+svm": 0.9636,
    "mitbih+knn": 0.9713}
	

d_train     = dataset['d']
t_train     = dataset['t']
start_cell  = dataset['start_cell']
best_metric = best_metric[f'{dataset_name}+{readout_name}']

# =======================
#  OPTIMIZATION PROBLEM
# =======================

def objective(sol):
	grid  = parse_grid(sol, start_cell)
	net   = generate_network(grid, start_cell.t_lims[1], start_cell.cell_type, len_d3=34)
	state = [ net.simulate(x, start_cell.t_lims[1]).spk_count() for x in d_train ]
	return -evaluate_kfold( state, t_train, model=readout)


# c = CSA(objective, pop_size=30, len_vec=80, n_process=30, AP=0.1, scale=0.05, n_mut=6)
csa = CSA(objective, pop_size=30, len_vec=80, n_process=30, AP=ap, scale=scale, n_mut=n_mut, max_grace=max_grace)

for _ in range(50):
    csa.evolve()
print( csa.g_best.fit/best_metric )