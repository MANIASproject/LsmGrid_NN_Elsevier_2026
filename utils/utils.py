import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation

from grids import Cell

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


from multiprocessing import Process, Manager
import json


# =============================
#      PLOTING FUNCTIONS
# =============================
def plot_grid2D(grid:list[Cell], color='blue', offset=0.5):
    for cell in grid:
        plt.vlines( [cell.t_lims[0], cell.t_lims[1]], cell.d2_lims[0]-offset, cell.d2_lims[1]-offset, color=color )
        plt.hlines( [cell.d2_lims[0]-offset, cell.d2_lims[1]-offset], cell.t_lims[0], cell.t_lims[1], color=color )
        
def plot_grid3D(grid:list[Cell], time:float, color='blue'):
	def in_time(cell, t):
		t0,t1 = cell.t_lims
		return t>=t0 and t<t1

	grid = [ cell for cell in grid if in_time(cell,time) ]

	for cell in grid:
		plt.vlines( [cell.d2_lims[0], cell.d2_lims[1]], cell.d3_lims[0], cell.d3_lims[1], color=color )
		plt.hlines( [cell.d3_lims[0], cell.d3_lims[1]], cell.d2_lims[0], cell.d2_lims[1], color=color )

def plot_number(instance):
    def update(i):
        im.set_data(frames[i])
        return [im]

    fig, ax = plt.subplots()
    frames = to_frame(instance)
    im = ax.imshow(frames[0], cmap='viridis', animated=True)

    ani = animation.FuncAnimation(fig,update, frames=len(frames), interval=100, blit=True)
    return HTML(ani.to_jshtml())

def to_frame(instance, time_window=10_000, frame_size=34):
    max = np.max([ t for t,(x,y) in instance ])
    frames = np.zeros( shape=(int(max/time_window)+1, frame_size,frame_size) )
    for t,(x,y) in instance:
        frames[int(t/time_window),y,x] = 1
    return frames

# =============================
#         EVALUATION
# =============================

def recall_mitbih(t_test, pred):
	t_test = np.array(t_test)
	pred   = np.array(pred)
	
	mask   = t_test!='N'
	recall = np.round( (pred[mask]==t_test[mask]).sum() / mask.sum(),4  )
	return recall

def evaluate(d_train, d_test, t_train, t_test, model, metric='accuracy'):
	scaler = StandardScaler().fit(d_train)
	d_train = scaler.transform(d_train)
	d_test  = scaler.transform(d_test)
	pred  = model.fit(d_train,t_train).predict(d_test)

	if metric=='accuracy'		:		return round( accuracy_score(t_test,pred), 4 )   
	elif metric=='recall_mitbih':		return recall_mitbih(t_test, pred)
	else						:		raise Exception('Valid metrics: accuracy, recall_mitbih')			

def evaluate_kfold(data,target, model, metric='accuracy'):
	kf = StratifiedKFold(n_splits=5)
	all_acc = []
	for idx_train, idx_test in kf.split(data,target):
		d_train = [ data[idx] for idx in idx_train ]
		d_test  = [ data[idx] for idx in idx_test ]
		t_train = [ target[idx] for idx in idx_train ]
		t_test  = [ target[idx] for idx in idx_test ]
		all_acc.append( evaluate(d_train,d_test,t_train,t_test, model, metric) )
	return np.mean(all_acc).round(4)


# ===========================
#     RESULTS MANAGMENT
# ===========================

class ResultsManager:
	def __init__(self, filename):
		self.filename = filename
		try:
			with open(filename, 'r') as file:
				self.results = json.load(file)
		except:
			self.results = {}
	def save(self):
		with open( self.filename, 'w' ) as file:
			json.dump( self.results, file )

	def add(self, key_name, trial):
		if not key_name in self.results:
			self.results[key_name] = []
		self.results[key_name].append(trial)
		self.save()
		

	def clear(self, key_name):
		confirm = input(('please type "clear"'))
		if confirm!='clear':
			return 'Cancelled'
		
		try:
			del self.results[key_name]
			with open( self.filename, 'w' ) as file:
				json.dump( self.results, file )
			return 'clear'
		except:
			return "key doesn't exsist"

# ===========================
#      METAHEURISTICS
# ===========================
class Solution:
	def __init__(self, sol, fit):
		self.sol = sol
		self.fit = fit


# ===========================
#      RESULT ANALYSES
# ===========================
def clean_grid(g_):
	g = []
	for c in g_:
		t_lims  = ( int(c['t_lims'][0]) , int(c['t_lims'][1])  )
		d2_lims = ( int(c['d2_lims'][0]), int(c['d2_lims'][1]) )
		try:
			d3_lims = ( int(c['d3_lims'][0]), int(c['d3_lims'][1]) )
		except:
			d3_lims = None

		g.append(Cell(t_lims=t_lims, d2_lims=d2_lims, d3_lims=d3_lims) )
	return g