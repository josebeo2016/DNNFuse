from .aasist_ssl.model import Model
import os
import logging
import numpy as np
import torch
from torch import nn
from torch import Tensor
import librosa
import gc

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename='running.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# model_path = "assist_ssl/pretrained/LA_model.pth" # old model
model_path = os.path.join(BASE_DIR, "aasist_ssl/pretrained/supcon_nov22_epoch_66.pth")
threshold = 1.47
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model(None, device).to(device)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(model_path,map_location=device), strict=False)
model.eval()


# def pad(x, max_len=64600):
#     x_len = x.shape[0]
#     if x_len >= max_len:
#         return x[:max_len]
#     # need to pad
#     num_repeats = int(max_len / x_len)+1
#     padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
#     return padded_x

def pad(x, padding_type='zero', max_len=64000):
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

def parse_input(file_path):
    cut = 64600  # take ~4 sec audio (64600 samples)
    X, fs = librosa.load(file_path, sr=16000)
    X_pad = pad(X, "repeat",cut)
    x_inp = Tensor(X_pad)
    return x_inp.unsqueeze(0).to(device)

def parse_input_sig(X, fs=16000, max_len=0, random_start=False, padding_type='zero'):
    '''
    X: audio signal
    fs: sampling rate, default 16000
    max_len: max length of the audio, default None - use full length
    random_start: if True, randomly pick a start point from 0 to len(X)-max_len if (len(X) > max_len)
    padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
    
    return: Tensor (1, max_len)
    
    '''
    cut = 64000  # take ~4 sec audio (64600 samples)
    if max_len == 0:
        # no cut, use full length
        X_pad = X
    else:
        if len(X) > max_len:
            if random_start:
                start = np.random.randint(0, len(X)-max_len)                    
                X_pad = X[start:start+max_len]
            else:
                X_pad = X[:max_len]
        else:
            # need to pad
            X_pad = pad(X, padding_type=padding_type, max_len=max_len)

    x_inp = Tensor(X_pad)
    return x_inp.unsqueeze(0).to(device)

def aasist_ssl_detect(wav):
    x_inp = parse_input(wav)
    out = model(x_inp)
    # per = nn.Softmax(dim=1)(out)
    return out[0][0].item(), out[0][1].item()

# def parse_input_sig(X, fs=16000):
#     cut = 64600  # take ~4 sec audio (64600 samples)
#     X_pad = pad(X, cut)
#     x_inp = Tensor(X_pad)
#     return x_inp.unsqueeze(0).to(device)

def aasist_ssl_detect_sig(y,sr=16000, **args):
    x_inp = parse_input_sig(y,sr, **args)
    out = model(x_inp)
    per = nn.Softmax(dim=1)(out)
    _, pred = out.max(dim=1)
    # return "spoof {0:.2f}%, bonafide {1:.2f}%".format(per[0][0].item()*100, per[0][1].item()*100)
    
    x_inp.cpu()
    del x_inp
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "spoof": per[0][0].item()*100, 
        "bonafide": per[0][1].item()*100
    }


def aasist_ssl_score_sig(y, sr=16000, sensitivity: float=1.0):
    x_inp = parse_input_sig(y,sr)
    out = model(x_inp)
    bona_score = out[0][1].item()
    if bona_score > (threshold + (sensitivity - 1)*(abs(threshold)/10)):
        return 1
    else:
        return 0

def aasist_ssl_score_chunk(ys: list, sr: int = 16000, sensitivity: float=1.0):
    """
    Get a list of audio chunks and return a average of the prediction
    """
    res = []
    for y in ys:
        res.append(aasist_ssl_score_sig(y, sr, sensitivity))
    return sum(res)/len(res)

def aasist_ssl_detect_chunk(ys: list, sr=16000):
    res = []
    for y in ys:
        res.append(aasist_ssl_detect_sig(y, sr))
    start = 0
    avg_spoof = sum([x["spoof"] for x in res])/len(res)
    
    # make the result more detailed
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
    


# print(assist_ssl_detect("/dataa/Dataset/ASVspoof/LA/ASVspoof2019_LA_eval/flac/LA_E_4581379.flac"))