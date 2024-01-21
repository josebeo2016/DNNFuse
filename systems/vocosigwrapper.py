from .modelwrapper import ModelWrapper
import gc
import logging
import numpy as np
import torch
from torch import nn
from torch import Tensor
import librosa
import yaml
from .scl_vocosig.model.wav2vec2_linear_nll import Model

# logging.basicConfig(filename='running.log', level=logging.DEBUG, format='[VocoSig] %(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
class VocoSigWrapper(ModelWrapper):
    def __init__(self, model_path:str, config_path:str, threshold:float):
        '''
        model_path: path to the trained model parameters file (pth)
        config_path: path to the model config file (yaml)
        threshold: threshold for the model to classify bonafide/spoof
        '''
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold
        self.model_path = model_path
        with open(config_path, 'r') as cfg:
            self.config = yaml.safe_load(cfg)
        
        self.model = Model(self.config['model'], self.device, is_train=False)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except:
            self.model = nn.DataParallel(self.model).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
    

    
    def parse_input(self, file_path:str) -> Tensor:
        '''
        extract necessary features from audio file
        file_path: path to the audio file
        '''
        # use parent class parse_input
        return super().parse_input(file_path)
    
    def detect(self, file_path:str) -> str:
        '''
        Detect bonafide/spoof from audio file
        file_path: path to the audio file
        return: string of the result
        '''
        # use parent class detect
        return super().detect(file_path)
    
    
    def parse_input_sig(self, X:np.ndarray, fs:int=16000, max_len:int=0, random_start:bool=False, padding_type:str='zero') -> Tensor:
        '''
        parse audio signal and extract necessary features for the model
        X: audio signal
        fs: sampling rate, default 16000
        max_len: max length of the audio, default None - use full length
        random_start: if True, randomly pick a start point from 0 to len(X)-max_len if (len(X) > max_len)
        padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
        
        '''
        # use parent class parse_input_sig
        return super().parse_input_sig(X, fs, max_len, random_start, padding_type)
    
    
    def detect_sig(self, y:np.ndarray, sr:int=16000, is_raw:bool=True, **args) -> dict:
        '''
        Detect bonafide/spoof from audio signal
        y: audio signal
        sr: sampling rate, default 16000
        is_raw: if True, return raw scores
        return: 
            if is_raw:
                list of raw scores - before calculate Softmax, dict of percentage
            else:
                dict of percentage
        '''
        # use parent class detect_sig
        return super().detect_sig(y, sr, is_raw, **args)
    
    def score_sig(self, y, sr=16000, sensitivity: float=1.0) -> int:
        '''
        Provide classification result from audio signal (1: bonafide, 0: spoof)
        '''
        # use parent class score_sig
        return super().score_sig(y, sr, sensitivity)
    
    def score_chunk(self, ys: list, sr: int = 16000, sensitivity: float=1.0) -> float:
        '''
        When the audio is too long, we need to split it into chunks and calculate the average score
        ys: list of audio signals
        sr: sampling rate, default 16000
        sensitivity: sensitivity of the model, default 1.0
        '''
        # use parent class score_chunk
        return super().score_chunk(ys, sr, sensitivity)
    
    def detect_chunk(self, ys: list, sr=16000) -> dict:
        '''
        When the audio is too long, we need to split it into chunks and calculate the average score
        ys: list of audio signals
        sr: sampling rate, default 16000
        return: dict of percentage
        '''
        # use parent class detect_chunk
        return super().detect_chunk(ys, sr)

# Example usage
if __name__ == "__main__":

    config_path = 'vocosig_model/configs/conf-5-linear.yaml'
    model_path = "vocosig_model/pretrained/conf-5-linear-nov22.pth"
    threshold = -4.4381

    vocosig_detector = VocoSigWrapper(config_path, model_path, threshold)
    score = vocosig_detector.vocosig_score_sig(y, sr)
    print(score)
