

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
    def __init__(self):
        '''
        initialize a data loader
        '''
        self.sourceFolder = '/mnt/storage/compling/proslearn/src/man_tone'
        self.dataset = 'test'
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
        try:
            with open(self.metapath,'rb') as f:
                self.metadata = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('meta file not found.')

        return self.metadata
    def load_data(self,datatype:str,amount:int,indexes = None):
        '''
        load certain amout of specific data
        :param datatype: should be in ['syllable','padded_wav','spectrogram','mel','highpass_spectrogram','lowpass_spectrogram','highpass_mel','lowpass_mel']
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

