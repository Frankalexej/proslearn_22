

import os
import pickle
import numpy as np
import pandas


class DataLoader():
    """
    a class for loading data from the source folder

    @Attributes:

    sourceFolder: /mnt/storage/compling/proslearn/src/eng

    typedict: ['spectrograms','mels','highpass_spectrograms','lowpass_spectrograms','highpass_mels','lowpass_mels']

    @Methods:

    get_metadata(metapath,datatype):
        load metadata and specify the data type that you want to load
    load_data(amount):
        randomly select certain amount of data

    """
    def __init__(self, sourceFolder='/mnt/storage/compling/proslearn/src/eng'):
        '''
        initialize a data loader
        '''
        self.sourceFolder = sourceFolder    # Frank 20241105: changed to configurable sourceFolder path
        self.typedict = ['spectrograms','mels','highpass_spectrograms','lowpass_spectrograms','highpass_mels','lowpass_mels']
        self.metapath = None
        self.metadata = None
        self.filepaths = None
        self.datasize = None
    def get_metadata(self,metapath:os.PathLike) -> pandas.DataFrame:
        '''
        initialize dataloader for certain type of data
        :param metapath: filepath for metadata, e.g. guide_test_syllableInfor.pkl
        :return: meta data
        '''
        self.metapath = os.path.join(self.sourceFolder, metapath)
        if self.metapath.endswith('.pkl'): 
            try:
                with open(self.metapath,'rb') as f:
                    self.metadata = pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError('meta file not found.')
        elif self.metapath.endswith('.csv'): 
            try: 
                self.metadata = pandas.read_csv(self.metapath)
            except FileNotFoundError: 
                raise FileNotFoundError('meta file not found.')

        return self.metadata
    
    def update_metadata(self,metadata:pandas.DataFrame):
        '''
        update metadata
        :param metadata: new metadata
        '''
        self.metadata = metadata

    def load_data(self,datatype:str,amount:int,indexes = None):
        '''
        load certain amout of specific data
        :param datatype: should be in ['syllable','spectrogram','mel','highpass_spectrogram','lowpass_spectrogram','highpass_mel','lowpass_mel']
        :param amount: number
        :param indexes: if not given, randomly generate one
        :return: a list of data, indexes in the metafile
        '''
        try:
            self.filepaths = self.metadata[datatype+"_path"]
            self.datasize = len(self.filepaths)
        except KeyError:
            raise KeyError('datatype not found.')
        if amount <= 0 or amount > self.datasize:
            raise ValueError(f'amount must be between 0 and {self.datasize}.')
        if indexes is None:
            indexes = np.random.choice(self.datasize,amount)
        selected_data = [np.load(f) for f in self.filepaths[indexes]]
        return selected_data,indexes



'''Sample

import pandas as pd
from librosa.util import index_to_slice
from scipy.signal import spectrogram

# meta data file (with syllable information)
import DataLoader as dl
import audioPreprocess as ap
import random
import dataPreprocess as dp

metafile = 'guide_test_syllableInfor.pkl'
datanum = 10000 # assume that you want 10000 tokens

suids,padded_waves = dp.load_dataset('/mnt/storage/compling/proslearn/src/eng/guide_test_waves_padded.pkl')
loader = dl.DataLoader()
meta = loader.get_metadata(metafile)
# you may randomly select certain number of data
mels,indexes = loader.load_data('mel',datanum)
spectrograms = loader.load_data('spectrogram',datanum,indexes)[0]
soundpaths = [meta['filepath'][i] for i in indexes]
# or you can get a bunch of indexlists that keeps the original distribution first
meta['index'] = meta.index
indexset = meta[['stress_type','index','suid']]
# Split into subsets
subsets_index = dp.split_into_subsets(indexset, num_subsets=50, sortkey='stress_type')
# #Display the resulting subsets
# for i, subset in enumerate(subsets):
#     dp.checkDistribution(subset['stress_type'])
seed = random.randint(0,datanum)
index = indexes[seed]
word = meta['word'][index]
syllble = meta['syllable'][index]
suid = meta['suid'][index]
stress_type = meta['stress_type'][index]
print(f'word:{word}; syllable: {syllble}; stress_type:{stress_type}; suid:{suid}')
sound,sr = ap.load_file(meta['filepath'][index])
ap.visualize_waveform(sound,sr)
padded = padded_waves[index]
ap.visualize_waveform(padded,sr)
ap.visualize_mel(mels[seed])
ap.visualize_spectrogram(spectrograms[seed])


'''