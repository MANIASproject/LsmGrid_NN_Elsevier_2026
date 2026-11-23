
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from copy import deepcopy

from grids import Partition, Splits,Cell, SpaceType
from utils import Solution
from multiprocessing import Pool


# =====================
#   RANDOM PARTITION
# =====================

def randomPartition(level,mode:SpaceType=SpaceType.D2):
	def recursive_split(p,level,mode):
		if level>0:
			pos   = np.round( np.random.uniform(0.2,0.8), 2 )
			n     = np.random.random()
			
			if mode==SpaceType.D2:
				split = Splits.T if n<0.5 else Splits.D2
			elif mode==SpaceType.D3:
				if n<0.33				:	split = Splits.T
				elif n>=0.33 and n<0.66	:	split = Splits.D2
				else					:	split = Splits.D3	
			else:
				raise Exception("Mode must be type SpaceType")

			p.assign(split,pos)
			recursive_split(p.left_child, level=level-1, mode=mode)
			recursive_split(p.right_child, level=level-1, mode=mode)
		else:
			p.assign( Splits.NULL )
		
	p = Partition(mode)
	recursive_split(p,level=level,mode=mode)
	return p

# =====================
#       HC-GP
# =====================
class HC_GP:
	def __init__(self, objective, cell:Cell, n_process=3, n_children=3, only_d1=False ):
		self.objective  = objective
		self.cell 	    = cell
		self.n_process  = n_process
		self.n_children = n_children
		self.mode       = cell.cell_type
		self.only_d1    = only_d1
		self.svc_start  = True           # kick start for the support vector classifier

		initial      = self.evaluate( [ randomPartition(level=2,mode=self.mode) for _ in range(n_children) ] )
		self.history = [ self.survive(initial) ]

	def evaluate(self, partitions:List[Partition]):
		""" Evaluate objective funtion and generate Solution Object """
		grids = [ p.get_grid(self.cell) for p in partitions ]
		with Pool(self.n_process) as pool:
			fitness = pool.map( self.objective, grids )
		return [Solution(p,fit) for p,fit in zip(partitions,fitness)]
	
	def survive(self,pop:List[Solution]):
		""" Only survive the best """
		return sorted( pop, key=lambda item:item.fit, reverse=False )[0]

	def mutate(self, sol:Partition):
		""" Create a new copy with one partition more """

		new_sol = deepcopy(sol)
		# Find a leaf
		levels, leaves = zip(*new_sol.find_leaves())
		levels = np.array(levels)
		levels = levels.max()-levels + levels.min()
		probs  = levels/levels.sum()
		leaf   = np.random.choice(leaves, p=probs)

		# Choice horizontal or vertical
		p  = np.round( np.random.uniform(0.2,0.8), 2 )
		n  = np.random.random()

		if self.only_d1:
			split = Splits.T
		elif self.mode==SpaceType.D2:
			split = Splits.T if n < 0.5 else Splits.D2
		elif self.mode==SpaceType.D3:
			if   n<0.33					: split = Splits.T
			elif n>=0.33 and n<0.66		: split = Splits.D2
			else						: split = Splits.D3
		
		leaf.assign(split, p, children_is_null=True)
		return new_sol
	
	def evolve(self, require_svc_start=False):

		# mitbih with svc requires more than 1 mutation to start giving improving solutions
		if require_svc_start and self.svc_start:
			new_sols  = [ self.mutate(self.history[-1].sol) for _ in range(self.n_children) ]
			for _ in range(2):
				new_sols = [ self.mutate(sol) for sol in new_sols ]
		else:
			new_sols  = [ self.mutate(self.history[-1].sol) for _ in range(self.n_children) ]

		candidato = self.survive( self.evaluate(new_sols) )
		if candidato.fit < self.history[-1].fit:
			self.history.append( candidato )
			self.svc_start = False
		else:
			self.history.append( self.history[-1] )

	def plot(self):
		plt.plot( [ sol.fit for sol in self.history ] )
		plt.grid()
		plt.title(f'g best {self.history[-1].fit}')