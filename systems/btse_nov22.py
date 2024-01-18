from .btse_model.model import Model
from .btse_model.biosegment import wav2bio

import os
from re import S
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
import yaml


threshold = -1.34
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "btse_model/configs/model_config_RawNet_Trans_64concat.yaml")
model_path = os.path.join(BASE_DIR, "btse_model/models/model_weighted_CCE_100_10_1e-06_supcon_nov22_Trans_64concat/epoch_25.pth") # nov22
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(config_path, 'r') as f_yaml:
    parser1 = yaml.safe_load(f_yaml)

model = Model(parser1['model'], device).to(device)
model = nn.DataParallel(model).to(device)

model.load_state_dict(torch.load(model_path,map_location=device))
model.eval()


def get_Bio(X_pad, fs):

    bio = wav2bio(X_pad, fs)
    # bio_length = len(bio)
    bio_inp = torch.IntTensor(bio)
    bio_length = torch.IntTensor([len(bio)])
    return bio_inp, bio_length

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def parse_input(file_path):
    cut = 64600  # take ~4 sec audio (64600 samples)
    X, fs = librosa.load(file_path, sr=16000)
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    bio_inp, bio_length = get_Bio(X_pad, fs)
    return x_inp.unsqueeze(0).to(device), bio_inp.unsqueeze(0).to(device), bio_length.to(device)

def btse_detect(wav):
    x_inp, bio_inp, bio_length = parse_input(wav)
    out, _ = model(x_inp, bio_inp, bio_length)
    return out[0][0].item(), out[0][1].item()


def parse_input_sig(X, fs=16000):
    cut = 64600  # take ~4 sec audio (64600 samples)
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    bio_inp, bio_length = get_Bio(X_pad, fs)
    return x_inp.unsqueeze(0).to(device), bio_inp.unsqueeze(0).to(device), bio_length.to(device)

def btse_detect_sig(y, sr=16000):
    x_inp, bio_inp, bio_length = parse_input_sig(y, sr)
    out, _ = model(x_inp, bio_inp, bio_length)
    per = nn.Softmax(dim=1)(out)
    _, pred = out.max(dim=1)
    # return "spoof {0:.2f}%, bonafide {1:.2f}%".format(per[0][0].item()*100, per[0][1].item()*100)
    return {
        "spoof": per[0][0].item()*100, 
        "bonafide": per[0][1].item()*100
        }
    
def btse_score_sig(y, sr=16000, sensitivity: float=1.0):
    x_inp, bio_inp, bio_length = parse_input_sig(y, sr)
    out, _ = model(x_inp, bio_inp, bio_length)
    if out[0][1].item() > (threshold + (sensitivity - 1)*(abs(threshold)/10)):
        return 1
    else:
        return 0

def btse_score_chunk(ys: list, sr: int = 16000, sensitivity: float=1.0):
    """
    Get a list of audio chunks and return a average of the prediction
    """
    res = []
    for y in ys:
        res.append(btse_score_sig(y, sr, sensitivity))
    return sum(res)/len(res)

# print(btse_detect("/dataa/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"))