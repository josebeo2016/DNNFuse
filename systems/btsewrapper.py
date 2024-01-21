from .modelwrapper import ModelWrapper
import gc
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
import yaml
import logging
from .btse_model.model import Model
from .btse_model.biosegment import wav2bio

# logging.basicConfig(filename = 'running.log', level = logging.DEBUG, format = '[BTSE] %(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# Create a new logging handler with your desired format
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(logging.DEBUG)
class BTSEWrapper(ModelWrapper):
    def __init__(self, model_path:str, 
                 config_path:str, 
                 threshold:int=-7.82):
        """
        model_path: path to the trained model parameters file (pth)
        config_path: path to the model config file (yaml)
        threshold: threshold for the model to classify bonafide/spoof
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.config_path = config_path
        self.threshold = threshold

        # Load config file
        with open(self.config_path, 'r') as f_yaml:
            self.config = yaml.safe_load(f_yaml)

        # Initialize and load model
        self.model = Model(self.config['model'], self.device).to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except:
            self.model = nn.DataParallel(self.model).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def get_Bio(self, X_pad:np.ndarray, fs:int = 16000) -> (Tensor, Tensor):
        '''
        extract Bio from audio signal
        X_pad: audio signal
        fs: sampling rate, default 16000
        '''
        bio = wav2bio(X_pad, fs)
        bio_inp = torch.IntTensor(bio)
        bio_length = torch.IntTensor([len(bio)])
        return bio_inp, bio_length

    def parse_input(self, file_path:str) -> (Tensor, Tensor, Tensor):
        '''
        Read audio file and extract necessary inputs for the model
        file_path: path to the audio file
        return: 
            x_inp: Tensor (1, #cut)
            bio_inp: Tensor (1, 1, #cut)
            bio_length: Tensor (1)
        '''
        cut = 64600 # take ~4 sec audio (64600 samples)
        X, fs = librosa.load(file_path, sr=16000)
        X_pad = self.pad(X, padding_type="zero", max_len=cut)
        x_inp = Tensor(X_pad)
        bio_inp, bio_length = self.get_Bio(X_pad, fs)
        return x_inp.unsqueeze(0).to(self.device), bio_inp.unsqueeze(0).to(self.device), bio_length.to(self.device)

    def detect(self, wav: str) -> str:
        '''
        Detect bonafide/spoof from audio file
        wav: path to the audio file
        return: string of the result
        '''
        
        out = None
        with torch.no_grad():
            x_inp, bio_inp, bio_length = self.parse_input(wav)
            out, _ = self.model(x_inp, bio_inp, bio_length)
        per = nn.Softmax(dim=1)(out)
        _, pred = out.max(dim=1)
        return out[0][0].item(), out[0][1].item()

    def parse_input_sig(self, X:np.ndarray, fs:int=16000, **args) -> (Tensor, Tensor, Tensor):
        '''
        Read audio signal and extract necessary inputs for the model
        X: audio signal
        fs: sampling rate, default 16000
        return:
            x_inp: Tensor (1, #cut)
            bio_inp: Tensor (1, 1, #cut)
            bio_length: Tensor (1)
        '''
        # cut = 64600  
        X_pad = self.pad(X, **args)
        x_inp = Tensor(X_pad)
        bio_inp, bio_length = self.get_Bio(X_pad, fs)
        return x_inp.unsqueeze(0).to(self.device), bio_inp.unsqueeze(0).to(self.device), bio_length.to(self.device)

    def detect_sig(self, y:np.ndarray, sr:int=16000, is_raw:bool = False, **args) -> dict:
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
        x_inp, bio_inp, bio_length = self.parse_input_sig(y, sr, **args)
        out = None
        with torch.no_grad():
            out, _ = self.model(x_inp, bio_inp, bio_length)
            per = nn.Softmax(dim=1)(out)
            _, pred = out.max(dim=1)
            logger.debug("raw score: {} {}".format(out[0][0].item(), out[0][1].item()))
        x_inp.cpu()
        del x_inp
        gc.collect()
        torch.cuda.empty_cache()
        if is_raw:
            return [out[0][0].item(), out[0][1].item()], {"spoof": per[0][0].item()*100, "bonafide": per[0][1].item()*100}
        return {"spoof": per[0][0].item()*100, "bonafide": per[0][1].item()*100}

    def score_sig(self, y:np.ndarray, sr:int=16000, sensitivity: float=1.0) -> int:
        '''
        Provide classification result from audio signal (1: bonafide, 0: spoof)
        '''
        x_inp, bio_inp, bio_length = self.parse_input_sig(y, sr)
        out, _ = self.model(x_inp, bio_inp, bio_length)
        if out[0][1].item() > (self.threshold + (sensitivity - 1)*(abs(self.threshold)/10)):
            return 1
        else:
            return 0  

    def score_chunk(self, ys: list, sr: int = 16000, sensitivity: float=1.0) -> float:
        '''
        When the audio is too long, we need to divide it into chunks and calculate the average score
        ys: list of audio signals
        sr: sampling rate, default 16000
        sensitivity: sensitivity of the model, default 1.0
        return: average score
        '''
        res = []
        for y in ys:
            res.append(self.score_sig(y, sr, sensitivity))
        return sum(res)/len(res)

    def detect_chunk(self, ys: list, sr:int=16000) -> dict:
        '''
        When the audio is too long, we need to divide it into chunks and calculate the average score
        ys: list of audio signals
        sr: sampling rate, default 16000
        return: dict of percentage
        '''
        res = []
        for y in ys:
            res.append(self.detect_sig(y, sr))
        avg_spoof = sum([x["spoof"] for x in res])/len(res)
        final = {
            "spoof": avg_spoof,
            "bonafide": 100.0 - avg_spoof,
            "detail": []
        }
        for i in range(len(res)):
            if i == len(res) - 1:
                final['detail'].append({'{}s-end'.format(i*4): res[i]})
            else:
                final['detail'].append({'{}s-{}s'.format(i*4,(i+1)*4): res[i]})
        return final


if __name__ == "__main__":
    detector = BTSEWrapper(model_path="btse_model/models/aug_jun22_tts_trans_concat64.pth", 
                 config_path="btse_model/configs/model_config_RawNet_Trans_64concat.yaml", 
                 threshold=-7.82)
    logger.info(detector.detect("/dataa/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"))