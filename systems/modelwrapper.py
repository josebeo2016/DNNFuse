# Abstract class for all model wrappers
import logging
import numpy as np
import torch
from torch import nn
from torch import Tensor
import librosa
import gc

logger = logging.getLogger(__name__)
# Create a new logging handler with your desired format
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(logging.DEBUG)
class ModelWrapper():
    def __init__(self) -> None:
        pass
    
    def pad(self, x:np.ndarray, padding_type:str='zero', max_len=64000, random_start=False) -> np.ndarray:
        '''
        pad audio signal to max_len
        x: audio signal
        padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
            zero: pad with zeros
            repeat: repeat the signal until it reaches max_len
        max_len: max length of the audio, default 64000
        random_start: if True, randomly choose the start point of the audio
        '''
        x_len = x.shape[0]
        logger.debug("x.shape: {}".format(x.shape))
        logger.debug("max_len: {}".format(max_len))
        # init padded_x
        padded_x = None
        if max_len == 0:
            # no padding
            padded_x = x
        elif max_len > 0:
            if x_len >= max_len:
                if random_start:
                    start = np.random.randint(0, x_len - max_len+1)
                    padded_x = x[start:start + max_len]
                    logger.debug("padded_x1: {}".format(padded_x.shape))
                else:
                    padded_x = x[:max_len]
                    logger.debug("padded_x2: {}".format(padded_x.shape))
            else:
                if padding_type == "repeat":
                    num_repeats = int(max_len / x_len) + 1
                    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
                    logger.debug("padded_x3: {}".format(padded_x.shape))
                elif padding_type == "zero":
                    padded_x = np.zeros(max_len)
                    padded_x[:x_len] = x
                    logger.debug("padded_x4: {}".format(padded_x.shape))
        else:
            raise ValueError("max_len must be >= 0")
        logger.debug("padded_x: {}".format(padded_x.shape))
        return padded_x
    
    def parse_input(self, file_path:str, **args) -> Tensor:
        '''
        extract necessary features from audio file
        file_path: path to the audio file
        padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
        '''
        X, fs = librosa.load(file_path, sr=16000)
        X_pad = self.pad(X, **args)
        x_inp = Tensor(X_pad)
        return x_inp.unsqueeze(0).to(self.device)
    
    def detect(self, file_path:str) -> str:
        '''
        Detect bonafide/spoof from audio file
        file_path: path to the audio file
        return: string of the result
        '''
        x_inp = self.parse_input(file_path)
        out = None
        with torch.no_grad():
            out = self.model(x_inp)
        per = nn.Softmax(dim=1)(out)
        _, pred = out.max(dim=1)
        return out[0][0].item(), out[0][1].item()
    
    def parse_input_sig(self, X:np.ndarray, fs:int=16000, max_len:int=0, random_start:bool=False, padding_type:str='zero') -> Tensor:
        '''
        parse audio signal and extract necessary features for the model
        X: audio signal
        fs: sampling rate, default 16000
        max_len: max length of the audio, default None - use full length
        random_start: if True, randomly pick a start point from 0 to len(X)-max_len if (len(X) > max_len)
        padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
        
        '''
        X_pad = self.pad(X, padding_type=padding_type, max_len=max_len, random_start=random_start)
        x_inp = Tensor(X_pad)
        return x_inp.unsqueeze(0).to(self.device)
    
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
        x_inp = self.parse_input_sig(y, fs=sr, **args)
        out = None
        with torch.no_grad():
            out = self.model(x_inp)
        # logger.debug("raw score: {} {}".format(out[0][0].item(), out[0][1].item()))
        per = nn.Softmax(dim=1)(out)
        _, pred = out.max(dim=1)
        # logger.debug("raw score: {} {}".format(out[0][0].item(), out[0][1].item()))
        x_inp.cpu()
        del x_inp
        gc.collect()
        torch.cuda.empty_cache()
        
        if is_raw:
            return [out[0][0].item(), out[0][1].item()], {"spoof": per[0][0].item()*100, "bonafide": per[0][1].item()*100}
        return {"spoof": per[0][0].item()*100, "bonafide": per[0][1].item()*100}
    
    def score_sig(self, y, sr=16000, sensitivity: float=1.0) -> int:
        '''
        Provide classification result from audio signal (1: bonafide, 0: spoof)
        '''
        x_inp = self.parse_input_sig(y, sr)
        out = self.model(x_inp)
        bona_score = out[0][1].item()
        
        if bona_score > (self.threshold + (sensitivity - 1) * (abs(self.threshold) / 10)):
            return 1
        else:
            return 0
    
    def score_chunk(self, ys: list, sr: int = 16000, sensitivity: float=1.0) -> float:
        '''
        When the audio is too long, we need to split it into chunks and calculate the average score
        ys: list of audio signals
        sr: sampling rate, default 16000
        sensitivity: sensitivity of the model, default 1.0
        '''
        res = []
        
        for y in ys:
            res.append(self.score_sig(y, sr, sensitivity))
        
        return sum(res) / len(res)
    
    def detect_chunk(self, ys: list, sr=16000) -> dict:
        '''
        When the audio is too long, we need to split it into chunks and calculate the average score
        ys: list of audio signals
        sr: sampling rate, default 16000
        return: dict of percentage
        '''
        res = []
        
        for y in ys:
            res.append(self.detect_sig(y, sr))
        
        start = 0
        avg_spoof = sum([x["spoof"] for x in res]) / len(res)
        final = {
            "spoof": avg_spoof,
            "bonafide": 100.0 - avg_spoof,
            "detail": []
        }
        
        for i in range(len(res)):
            if i == len(res) - 1:
                final['detail'].append({'{}s-end'.format(i * 4): res[i]})
            else:
                final['detail'].append({'{}s-{}s'.format(i * 4, (i + 1) * 4): res[i]})
        
        return final
    
    def detect_bayesian_sig(self, y:np.ndarray, sr:int=16000, is_raw:bool=True, number_trails: int=10, **args) -> dict:
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
        # random start and make different padding for each trail
        x_inp = []
        for i in range(number_trails):
            x_inp.append(self.parse_input_sig(y, fs=sr, **args).squeeze(0))
        # convert x_inp array into torch tensor by stacking
        x_inp = torch.stack(x_inp,dim=0)
        logger.debug("x_inp: {}".format(x_inp.shape))
        # predict
        # x_inp = self.parse_input_sig(y, fs=sr, **args)
        out = None
        with torch.no_grad():
            out = self.model(x_inp)
        # logger.debug("out: {}".format(out))
        
        # calculate mean and std
        # logger.debug("Calculate mean of {} trails".format(number_trails))
        mean = torch.mean(out, dim=0)
        # std = torch.std(out, dim=0)
        
        # calculate softmax
        per = nn.Softmax(dim=1)(out)
        logger.debug("per: {}".format(per))
        # calculate mean probability
        mean_prob = torch.mean(per, dim=0)
        # logger.debug("raw score: {} {}".format(out[0][0].item(), out[0][1].item()))
        x_inp.cpu()
        del x_inp
        gc.collect()
        torch.cuda.empty_cache()
    
        if is_raw:
            return [mean[0].item(), mean[1].item()], {"spoof": mean_prob[0].item()*100, "bonafide": mean_prob[1].item()*100}
        return {"spoof": mean_prob[0].item()*100, "bonafide": mean_prob[1].item()*100}