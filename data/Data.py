import tonic
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle

from tqdm import tqdm
import pickle
from abc import ABC, abstractmethod

import os
import librosa


class Dataset(ABC):
	@abstractmethod
	def preProcessAndSave(self, random_state):
		""" Preprocess the data and save it ready for use """
		pass
	def load(self):
		with open(self.file_route,'rb') as file:
			dataset = pickle.load(file)
		self.d_train = dataset['d_train']
		self.d_test  = dataset['d_test']
		self.t_train = dataset['t_train']
		self.t_test  = dataset['t_test']

		
class NTIDIGITS(Dataset):
	def __init__(self):
		self.file_route = './data/tidigits_processed.pickle'
		self.d_train = None
		self.d_test  = None
		self.t_train = None
		self.t_test  = None

	def findMainSignal(self,instance, k=20, th=0.1, goodness_right=0.01, th_up=55, th_bottom=10, plot=False):
		max_t  = 1_500
		offset = 100 # Move the entire graph to the right so that the convolution is not lost if the signal is too close to the left edge

		# ADD ACTIVITY
		only_time = np.array([ 0 for _ in range(max_t) ])
		for t,idx in instance:
			if int(t)+offset<max_t and idx<th_up and idx>th_bottom:
				only_time[ int(t)+offset ] += 1

		# CONVOLUTION
		conv = []
		for i in range(k,max_t-k):
			t = i
			c = 0.01*only_time[t-k:t+k].sum()
			conv.append(c)
		conv = np.concatenate( (np.zeros(k), conv, np.zeros(k)) )

		# FIND ALL PEAKS
		bloques = []
		state   = 'DOWN'
		for t in range(0,max_t):
			if state=='DOWN' and conv[t]>th:
				new_start = t
				state     = 'UP'
			elif state=='UP' and conv[t]<th:
				bloques.append( ( conv[new_start:t].max(), new_start,t) )
				state = 'DOWN'
		_,start,end = sorted( bloques, key=lambda item: item[0]*(item[2]-item[1]), reverse=True )[0]

		# GOODNESS OF THE RIGHT DOWNSLOPE
		for end in range(end,max_t):
			if conv[end]<goodness_right:	break

		return start-offset-15,end-offset
			
	def extractMainSignal(self, instance, start, end):
		return [ (time-start,channel) for time,channel in instance if time>=start and time<=end]
	
	def us_to_ms(self, instance):
		return [ (time/1000,idx) for time,idx,_ in instance ]

	def preProcessAndSave( self, random_state, k=5):
		# LOAD DATA
		d_train = tonic.datasets.NTIDIGITS18(save_to="./data", train=True, single_digits=True)
		d_test  = tonic.datasets.NTIDIGITS18(save_to="./data", train=False, single_digits=True)
		d_train,t_train = zip(*d_train)
		d_test ,t_test  = zip(*d_test)

		# PREPROCESS
		""" Preprocess: Convert usec to msec, extract main signal, downsampling """

		rng = np.random.RandomState(random_state)
		d_train_processed	 = []
		for instance in tqdm(d_train):
			instance   = self.us_to_ms(instance)
			instance   = [ (float(t), int(idx)) for t,idx in instance ]
			start, end = self.findMainSignal(instance, k=7, th=0.1, goodness_right=0.01)
			instance   = self.extractMainSignal(instance,start,end)
			instance   = [ spike for spike in instance if rng.random()<0.7 ]
			d_train_processed.append( instance )

		d_test_processed	 = []
		for instance in tqdm(d_test):
			instance   = self.us_to_ms(instance)
			instance   = [ (float(t), int(idx)) for t,idx in instance ]
			start, end = self.findMainSignal(instance, k=7, th=0.1, goodness_right=0.01)
			instance   = self.extractMainSignal(instance,start,end)
			instance   = [ spike for spike in instance if rng.random()<0.7 ]
			d_test_processed.append( instance )

		# SAVE
		dataset = {'d_train':d_train_processed, 'd_test':d_test_processed, 
			 		't_train':t_train, 't_test':t_test}
		
		with open(self.file_route,'wb') as file:
			pickle.dump( dataset,file )
		
		self.d_train = dataset['d_train']
		self.d_test  = dataset['d_test']
		self.t_train = dataset['t_train']
		self.t_test  = dataset['t_test']

class FSDD(Dataset):
    def __init__(self, time_bin=40, max_spikes=10, n_dim=39, hop_lenght=300, file_route='./data/fsdd_processed.pickle'):
        self.n_dim 	    = n_dim
        self.time_bin   = time_bin
        self.max_spikes = max_spikes
        self.hop_lenght = hop_lenght

        self.file_route = file_route
        self.d_train = None
        self.d_test  = None
        self.t_train = None
        self.t_test  = None

    def poisson_process( self, n_spikes , t_start, t_end, seed):
        """ n_spikes = expected spikes per time unit!! """
        rng = np.random.default_rng(seed)
        rate = n_spikes/(t_end-t_start)
        spikes = []
        if n_spikes==0:
                return spikes
        t = t_start
        while True:
            isi = rng.exponential( 1/rate )
            t += isi
            if t<t_end: spikes.append( t )
            else      : break
        return spikes
    
    def encodeAudio(self,y,sr):
        bins = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=self.n_dim, hop_length=self.hop_lenght, n_fft=2*self.hop_lenght)
        return bins

    def normalizeData(self,data_train, data_test):				# data debe tener forma (samples, fetaures, time)
        mins = data_train.min(axis=(0,2), keepdims=True)
        maxs = data_train.max(axis=(0,2), keepdims=True)
        data_train = (data_train - mins) / (maxs - mins + 1e-8)
        data_test  = (data_test  - mins) / (maxs - mins + 1e-8)

        data_test = np.clip(data_test, 0, 1)
        return data_train, data_test

    def binFrecsToSpikeSeries( self, instance, time_bin, max_spikes, seed ):
        """
        Sample  : bins with frequencies with shape (channels, timesteps). The timesteps represent 
                    time bins that indicate frequencies, where 1 is the maximum frequency 
        time_bin: How much time each bin will be simulated
        dt		: The delta-time of the simulation
        """
        assert instance.max()<=1,'Los instances deben venir escalados'

        instance_num_spikes = (instance*max_spikes).astype(np.int32)	# here sample has shape (channels, num_spikes per bin)
        instance_spikes     = []										# sample expressed in spikes

        dt = 0.1	
        for channel_idx, channel in enumerate(instance_num_spikes):	# each channel here is a time serie
            for bin_idx, bin_num_spikes in enumerate(channel):		# each bin is a value that represents the amount of spikes in that time bin
                inicio_bin = bin_idx*time_bin
                fin_bin    = (bin_idx+1)*time_bin-dt
                spikes 	   = np.linspace(inicio_bin,fin_bin,bin_num_spikes,dtype=np.float32) + np.random.normal(0,0.5)
                spikes     = np.clip(spikes,0,1_000)
                # spikes 	   = self.poisson_process(bin_num_spikes,inicio_bin, fin_bin, seed) # exact spike times
                for spike_time in spikes: 
                    instance_spikes.append( (float(spike_time), channel_idx) )

        return sorted(instance_spikes, key=lambda item:item[0])

    def preProcessAndSave(self, random_state, record_folder='./data/recordings'):
        # CARGAR DATOS
        files = os.listdir(record_folder)
        data    = []
        target  = []
        for file in tqdm(files):
            y, sr = librosa.load(f'{record_folder}/{file}', sr=8_000)
            y = librosa.util.fix_length(y, size=3000)
            bins = self.encodeAudio(y,sr) 
            data.append( bins )
            target.append( int(file.split('_')[0]) )

        d_train,d_test, t_train, t_test = train_test_split(np.array( data ),np.array(target), train_size=0.8, shuffle=True, stratify=target, random_state=random_state)
        
        # PREPROCESS TRAIN TEST
        d_train, d_test   = self.normalizeData( d_train, d_test)
        d_train_processed = [ self.binFrecsToSpikeSeries( instance, self.time_bin, self.max_spikes, random_state ) for instance in tqdm(d_train)]
        d_test_processed  = [ self.binFrecsToSpikeSeries( instance, self.time_bin, self.max_spikes, random_state ) for instance in tqdm(d_test)]

        # SAVE
        dataset = {'d_train':d_train_processed, 'd_test':d_test_processed, 
                    't_train':t_train, 't_test':t_test}
        
        with open(self.file_route,'wb') as file:
            pickle.dump( dataset,file )		

        self.d_train = dataset['d_train']
        self.d_test  = dataset['d_test']
        self.t_train = dataset['t_train']
        self.t_test  = dataset['t_test']

class NMNIST(Dataset):
    def __init__(self, file_route='./data/nmnist_processed.pickle'):
        self.file_route = file_route
        self.d_train = None
        self.d_test  = None
        self.t_train = None
        self.t_test  = None

    def getSubset(self,data,size,random_state):
        all_targets = [ t for d,t in tqdm(data) ]
        d = []
        t = []

        rng = np.random.default_rng(seed=random_state)
        for label in range(10):
            all_idx = rng.choice( [ idx for idx,t in enumerate(all_targets) if t==label ], size=size//10, replace=False )
            temp = [ data[idx] for idx in all_idx ]
            temp_d, temp_t = zip(*temp) 
            d += temp_d
            t += temp_t
        return d,t

    def preprocess(self,d):
        new_d = [None for _ in range(len(d))]
        drop = tonic.transforms.DropEvent(p=0.1)
        for idx in tqdm(range(len(d))):
            instance   = np.array([e for e in d[idx] if e[3]==0 ])        # drop polarity
            instance   = drop(instance)                                   # drop random events
            instance   = [ (t/1000,(x,y)) for x,y,t,p in instance ]       # as np.int64 array and time in ms
            new_d[idx] = instance
        return new_d

    def preProcessAndSave(self, random_state, len_subset=5_000, k=5):

        # LOAD DATA
        data_train = tonic.datasets.NMNIST(save_to='data', train=True)
        data_test  = tonic.datasets.NMNIST(save_to='data', train=False)
        
        # SELECT SUBSET
        len_train = int(0.8*len_subset)
        len_test  = len_subset-len_train
        d_train,t_train = self.getSubset(data_train, size=len_train, random_state=random_state)
        d_test, t_test  = self.getSubset(data_test,  size=len_test,  random_state=random_state) 

        # PREPROCESS
        d_train = self.preprocess(d_train)
        d_test  = self.preprocess(d_test)     

        d_train, t_train = shuffle( d_train,t_train )   
        d_test , t_test  = shuffle( d_test ,t_test  )   

        # SAVE
        dataset = {'d_train':d_train, 'd_test':d_test, 
                    't_train':t_train, 't_test':t_test}
        
        with open(self.file_route,'wb') as file:
            pickle.dump( dataset,file )
        
        self.d_train = dataset['d_train']
        self.d_test  = dataset['d_test']
        self.t_train = dataset['t_train']
        self.t_test  = dataset['t_test']
		
class MITBIH(Dataset):
    def __init__(self):
        self.file_route = './data/mitbih_processed.pickle'
        self.folder     = './data/mit-bih'
        self.d_train = None
        self.d_test  = None
        self.t_train = None
        self.t_test  = None

    # ===============
    #   OPEN FILES
    # ===============
    def open_annotations( self, ID):
        not_useful = {'+','~','!','"','[',']','x','|'}
        annotations_map = {'/':'Q','A':'S','E':'V','F':'F','J':'S','L':'N','N':'N','Q':'Q','R':'N','S':'S','V':'V','a':'S','e':'N','f':'Q','j':'N'}

        def filter_line(line):
            line = line.strip().split()
            return [ col for col in line if col != '' ][:3]

        with open(f'{self.folder}/{ID}annotations.txt','r') as file:
            file = file.readlines()

        annotations = [filter_line(line) for line in file[1:]]
        annotations = pd.DataFrame( data=annotations, columns=['ts','sample','type'] )
        annotations['sample'] = annotations['sample'].astype(np.int64)
        
        annotations = annotations[~annotations['type'].isin(not_useful)]
        annotations['type'] = annotations['type'].map( lambda row: annotations_map[row] )
        return annotations
    
    def open_files(self, ID):
        signal = pd.read_csv(f'{self.folder}/{ID}.csv')
        signal.columns = ['sample','MLII','V5']
        signal = signal[['sample','MLII']]
        signal = signal.set_index('sample')
        anno = self.open_annotations(ID)
        return signal, anno
    
    def extract_samples( self, id, padding = 146):
        signals = []
        labels  = []  
        ecg_signal, annotations = self.open_files(id)

        for row in annotations.itertuples():
            center = row.sample
            type   = row.type
            signal = ecg_signal.loc[center-padding:center+padding]
            if len(signal)==2*padding+1:
                signals.append( signal )
                labels.append( type )

        signals = np.array( signals ).reshape(-1,2*padding+1)
        return signals,labels
    
    # ===============
    #  PREPROCESSING
    # ===============
    def delta_modulation( self, signal, delta_value):
        channel_neg = 0
        channel_pos = 1
        spikes = []
        reference = signal[0]
        for t,signal_value in enumerate(signal):
            t += np.round( np.random.normal(loc=0, scale=0.1), 2 )
            if signal_value > reference+delta_value:
                spikes.append( (t,channel_pos) )
                reference = signal_value
            elif signal_value < reference-delta_value:
                spikes.append( (t,channel_neg) )
                reference = signal_value
        return spikes
    
    def normalize(self, beat):
        mu  = np.mean(beat)
        std = np.std(beat)
        return (beat-mu)/std

    def preProcessAndSave(self, random_state=88, delta_value=0.3):

        # OPEN ALL FILES
        excluded = {'102.csv', '104.csv', '107.csv', '217.csv'}
        ids = [ filename.split('.')[0] for filename in os.listdir(self.folder) if ('csv' in filename) and (not filename in excluded) ]

        all_signals,all_labels = [], []
        for id in tqdm(ids):
            signals,labels  = self.extract_samples(id)
            all_signals.append( signals )
            all_labels.append( labels )

        signals = np.concatenate( all_signals, axis=0 )
        labels  = np.concatenate( all_labels,axis=0 )

        # SPLIT AND SCALE DATA
        signals = [self.normalize(s) for s in signals ]
        signals_train,signals_test, labels_train,labels_test =  train_test_split(signals, labels, random_state=random_state, stratify=labels, train_size=0.8)


        # DROP SOME NORMAL DATA
        #   IN THE TRAIN SET
        all_idx_n       = [ idx for idx,label in enumerate(labels_train) if label=='N' ]
        n_labels_v      = (labels_train=='V').sum()
        preserve_idx_n  = set(np.random.choice( all_idx_n, n_labels_v, replace=False ) )
        preserve_id_all = [ idx for idx, label in enumerate(labels_train) if (label!='N') or (idx in preserve_idx_n) ]
        labels_train    = [ labels_train[idx] for idx in preserve_id_all ]
        signals_train   = [ signals_train[idx] for idx in preserve_id_all ]

        # CONVERT TO SPIKES
        d_train = [ self.delta_modulation(s, delta_value) for s in tqdm(signals_train) ]
        d_test  = [ self.delta_modulation(s, delta_value) for s in tqdm(signals_test)  ]

     
        # SAVE
        dataset = {'d_train':d_train, 'd_test':d_test, 
                    't_train':labels_train, 't_test':labels_test}
        
        with open(self.file_route,'wb') as file:
            pickle.dump( dataset,file )		

        self.d_train = dataset['d_train']
        self.d_test  = dataset['d_test']
        self.t_train = dataset['t_train']
        self.t_test  = dataset['t_test']
    

