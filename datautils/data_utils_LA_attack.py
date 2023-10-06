import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from .RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

types = {
    "-": 0,
    "A01": 1,
    "A02": 2,
    "A03": 3,
    "A04": 4,
    "A05": 5,
    "A06": 6,
    "A07": 7,
    "A08": 8,
    "A09": 9,
    "A10": 10,
    "A11": 11,
    "A12": 12,
    "A13": 13,
    "A14": 14,
    "A15": 15,
    "A16": 16,
    "A17": 17,
    "A18": 18,
    "A19": 19,
}

def genList(dir_meta,is_train=False,is_eval=False, is_dev=False):
    """
    dir_meta: string, path to the meta file, meta file format: utt_id subset category label
    """
    
    labels = {}
    categories = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            key = line.strip().split()
            if (key[1]=='dev'): # reverse train and dev for avoiding overfitting
                file_list.append(key[0])
                labels[key[0]] = 1 if key[3] == 'bonafide' else 0
                categories[key[0]] = types[key[2]]
        return labels, categories, file_list
    elif (is_dev):
        for line in l_meta:
            key = line.strip().split()
            if (key[1]=='train'): # reverse train and dev for avoiding overfitting
                file_list.append(key[0])
                labels[key[0]] = 1 if key[3] == 'bonafide' else 0
                categories[key[0]] = types[key[2]]
        return labels, categories, file_list
    elif (is_eval):
        for line in l_meta:
            key = line.strip().split()
            if (key[1]=='eval'):
                file_list.append(key[0])
        return file_list

def pad(x, padding_type, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    if padding_type == "repeat":
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    elif padding_type == "zero":
        padded_x = np.zeros(max_len)
        padded_x[:x_len] = x
    return padded_x
			

class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, categories, base_dir, algo, cm_scores=None):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
            
        self.list_IDs = list_IDs
        self.labels = labels
        self.categories = categories
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cm_scores = cm_scores
        self.cut=64600 # take ~4 sec audio (64600 samples)
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X,fs = librosa.load(os.path.join(self.base_dir, utt_id), sr=16000) 
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_pad= pad(Y,"zero",self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        category = self.categories[utt_id]
        
        # get score from CM system
        if self.cm_scores is not None:
            scores = self.cm_scores(utt_id, data_dir=self.base_dir) # (2, #number CM)

        return x_inp, scores, target, category
            
            
class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cm_scores=None):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)
        self.cm_scores = cm_scores
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, utt_id), sr=16000)
        X_pad = pad(X,"zero",self.cut)
        x_inp = Tensor(X_pad)
        # get score from CM system
        if self.cm_scores is not None:
            scores = self.cm_scores(utt_id, data_dir=self.base_dir) # (2, #number CM)
        return x_inp, scores, utt_id




#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
