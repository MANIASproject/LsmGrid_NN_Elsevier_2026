import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from utils import Solution
from grids import Splits, Partition, SpaceType,Cell

# ==============
#    PARSER 
# ==============
""" The parser convert a real vector into a grid """

def parse_grid(chromosome, cell:Cell, only_d1=False, initial_partition=None):
	chromosome = np.clip( np.round(chromosome,2),0,1 )

	n_codons    = np.floor( len(chromosome)/2 ).astype(np.int64)
	if only_d1	:	split_types = [Splits.NULL, Splits.T]
	else		:	split_types = [Splits.T, Splits.D2, Splits.D3]
	n_options   = 2 if (cell.cell_type==SpaceType.D2 or only_d1) else 3

	p = Partition( cell.cell_type ) if initial_partition is None else initial_partition
	all_undef = p.get_undef()

	for i in range(0,n_codons):
		codon0 = chromosome[2*i]
		codon1 = chromosome[2*i+1]
		
		split_type = split_types[ (n_options*codon0).astype(np.int64)%n_options ]
		split_p    = np.round(0.2+0.6*codon1,3)
		
		if len(all_undef)==0:
			break

		node = all_undef.pop(0)
		node.assign( split_type, split_p )
		all_undef.append( node.left_child  )
		all_undef.append( node.right_child )

	for node in all_undef:
		node.assign( Splits.NULL )

	return p.get_grid( cell )


# =================
#   BaseOptimizer
# =================
""" Basic functions common in many metaheuristics """
class BaseOptimizer:
	def __init__(self, objective, pop_size, len_vec, n_process, repare=None):
		self.objective = objective
		self.pop_size  = pop_size
		self.len_vec   = len_vec
		self.n_process = n_process
		self.repare    = repare

		self.pop     = self.evaluate( [ self.generate_random_solution() for _ in range(pop_size) ] )
		self.g_best  = sorted( self.pop, key=lambda item:item.fit )[0]
		self.history = [ self.g_best ]

	# ===========
	#  METRICS
	def diversity(self, pop, n_options):
		topologies = [ c.sol.reshape(-1,2)[:,0] for c in pop ]
		topologies = [ tuple((t*n_options).astype(np.int64)) for t in topologies ]
		return len(set(topologies))

	def plot(self):
		plt.plot( [-sol.fit for sol in self.history] )

	# ===========
	#  BASIC MH
	def generate_random_solution(self):
			return np.random.random( self.len_vec )

	def evaluate(self, pop):
		with Pool(processes=self.n_process) as pool:
			fitness = pool.map( self.objective, pop )
		return [ Solution(sol,fit) for sol,fit in zip(pop,fitness) ]

	# =====================
	#   USEFULL OPERATORS
	def gaussian_noise(self, sol, scale):
		sol = sol.reshape(-1,2)
		sol_discrete  = sol[:,0]
		sol_continuos = np.random.normal( loc=sol[:,1], scale=scale )
		sol = [ (d,c) for d,c in zip(sol_discrete, sol_continuos) ]
		return np.array(sol).reshape(-1).clip(0,1)

	def mutate_frame(self, sol, n_mut):
		sol = sol.reshape(-1,2).copy()
		sol_discrete  = sol[:,0]
		sol_continuos = sol[:,1]
		pos = np.random.choice(len(sol_discrete), n_mut, replace=False)

		for p in pos:
			sol_discrete[p] = np.random.random()
		sol = [ (d,c) for d,c in zip(sol_discrete, sol_continuos) ]
		return np.array(sol).reshape(-1)


# =================
#       CSA
# =================
""" Adapted version of Crow Seatch Algorithm """
class CSA(BaseOptimizer):
	def __init__(self, *args, scale=0.05, AP=0.24, n_mut=3, max_grace=1):
		super().__init__(*args)

		# csa parameters
		self.scale     = scale
		self.AP        = AP
		self.n_mut     = n_mut
		self.max_grace = max_grace
		
		# csa mechanisms
		self.c_best = self.pop 
		self.grace  = np.zeros(self.pop_size)
		self.stats  = [] 
	
	def tricked(self,crow):
		return self.mutate_frame(crow, self.n_mut)

	def follow(self,crow):
		return self.gaussian_noise(crow, self.scale)

	def evolve(self):
		operations = []
		new_pop    = []
		for c1 in range(self.pop_size):
			if self.grace[c1]:
				new_pop.append( self.follow( self.pop[c1].sol ) )
				self.grace[c1] -= 1
				operations.append('refine')
			else:
				c2 = np.random.choice( [idx for idx in range(self.pop_size) if idx!=c1] )   # random crow
				c2 = self.c_best[c2].sol
				# random
				if np.random.random()<self.AP:
					new_pop.append( self.tricked(c2) )
					self.grace[c1] = self.max_grace
					operations.append('tricked')
				# follow
				else:
					new_pop.append( self.follow(c2) )
					operations.append('follow')
		self.pop = self.evaluate(new_pop)  

		stats = {'follow':0, 'tricked':0, 'refine':0}
		# UPDATE C_BEST AND G_BEST
		self.imp_reason = None
		for c in range(self.pop_size):
			if self.pop[c].fit <= self.c_best[c].fit:
				self.c_best[c] = self.pop[c]
				stats[operations[c]] += 1
			
			if self.pop[c].fit <= self.g_best.fit:
				self.g_best = self.pop[c]
				self.imp_reason = operations[c]
		
		self.history.append( self.g_best )
		self.stats.append( stats )


# =================
#        GA
# =================
""" Adapted version of a Genetic Algorithm """
class GA(BaseOptimizer):
	def __init__(self, *args, scale, n_mut, p_mut_frame, tournament_k):
		super().__init__(*args)
		self.scale        = scale
		self.n_mut        = n_mut
		self.tournament_k = tournament_k
		self.p_mut_frame  = p_mut_frame

	def mut_continuos(self, sol):
		return self.gaussian_noise(sol, self.scale)

	def mut_discrete(self, sol):
		return self.mutate_frame(sol, self.n_mut)

	def crossover(self, parent_a, parent_b):
		mask = 1*np.repeat( np.random.random(self.len_vec//2)<0.5, 2 )
		child1 = parent_a*mask + parent_b*(1-mask)
		child2 = parent_b*mask + parent_a*(1-mask)
		return child1, child2

	def survive(self, pop):
		return sorted( pop, key=lambda item:item.fit )[0:self.pop_size]

	def tournament(self, exclude=None):
		while True:
			candidates = np.random.choice( self.pop_size, self.tournament_k, replace=False )
			best = self.pop[ candidates[0] ]
			for c in candidates:
				if self.pop[c].fit<=best.fit:
					best = self.pop[c]
			if best!=exclude:
				return best

	def evolve(self):
		new_pop = []
		for _ in range(self.pop_size//2):
			# selection
			p1 = self.tournament()
			p2 = self.tournament(p1)

			# crossover
			c1,c2 = self.crossover(p1.sol,p2.sol)

			# mutation
			if np.random.random() < self.p_mut_frame:	c1 = self.mut_discrete(c1)
			else 									: 	c1 = self.mut_continuos(c1)
			if np.random.random() < self.p_mut_frame:	c2 = self.mut_discrete(c2)
			else 									: 	c2 = self.mut_continuos(c2)
			new_pop += [c1,c2]

		# survive
		new_pop = self.evaluate(new_pop)
		self.pop = self.survive( self.pop+new_pop )

		# update
		if self.pop[0].fit <= self.g_best.fit:
			self.g_best = self.pop[0]
		self.history.append( self.g_best )
	
	def plot(self):
		plt.plot( [-sol.fit for sol in self.history] )	