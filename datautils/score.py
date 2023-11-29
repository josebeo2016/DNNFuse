import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .systems import CMS
import os
import pandas as pd
import time
# dynamic load detection function from each CM
class Score:
    def __init__(self, score_path: str = "scores/", CMS: list = ['aasist_ssl', 'vocosig']) -> None:
        pass
        self.cm_mode = {}
        self.cm_df = {}
        self.CMS = CMS

        for cm in self.CMS:
            exec("from systems import %s_detect" % cm)
            self.cm_mode[cm] = 'online' # or offline
            
        # check score path exists and set mode
        for cm in self.CMS:
            if not os.path.exists(os.path.join(score_path, cm + ".txt")):
                print("Score file for %s does not exist!" % cm)
                print("Switch to online calculation mode")
                self.cm_mode[cm] = 'online'
            else:
                self.cm_mode[cm] = 'offline'
                self.cm_df[cm] = pd.read_csv(os.path.join(score_path, cm + ".txt"), sep=" ", header=None)
                self.cm_df[cm].columns = ["utt_id", "spoof", "bonafide"]
                # make index column
                self.cm_df[cm]["index"] = self.cm_df[cm]['utt_id'].apply(lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
                self.cm_df[cm].set_index("index", inplace=True)
                # print(self.cm_df[cm].head)

        print("Finished loading score function")

    def get_cm_scores(self, utt_id, data_dir=None):
        """
        utt_id: string, the utterance id
        """
        # load scores
        cm_scores = []
        for cm in self.CMS:
            if (self.cm_mode[cm] == 'offline') and len(self.cm_df[cm][self.cm_df[cm]["utt_id"] == utt_id])>0:
                # get index:
                index = utt_id.split("/")[-1].split(".")[0].split("_")[-1]
                df = self.cm_df[cm]
                # cm_scores.append(df[df["utt_id"] == utt_id][["spoof", "bonafide"]].values[0])
                scors = df.loc[[int(index)]][["spoof", "bonafide"]].values[0]
                # append to cm_scores
                cm_scores.append(scors)
            else:
                print("Calculate %s score online for %s" % (cm, utt_id))
                exec("cm_scores.append(%s_detect('%s'))" % (cm, os.path.join(data_dir, utt_id)))
        # convert to np array
        # print(cm_scores.shape)
        cm_scores = np.array(cm_scores)
        scores_tensor = torch.Tensor(cm_scores).transpose(0,1) # (2, #number CM)
        # print(scores_tensor)
        assert scores_tensor.shape == (2, len(self.CMS))
        return  scores_tensor

if __name__ == '__main__':
    start_time = time.time()
    score_cls = Score(score_path="scores/la2019/")
    score_tensor = score_cls.get_cm_scores("eval/LA_E_8877452.flac", data_dir="DATA/asvspoof2019/")
    print("score_tensor:", score_tensor)
    print("first time: ", time.time() - start_time)
    start_time = time.time()
    # weight = torch.Tensor([0.7, 0.3])
    
    # print("softmax:", F.softmax(score_tensor, dim=0))
    
    # print("matmul:", torch.matmul(F.softmax(score_tensor, dim=0), weight))
    # print("sum:", torch.sum(F.softmax(score_tensor, dim=0) * weight, dim=0))
    
    score_tensor = score_cls.get_cm_scores("train/LA_T_9242200.flac", data_dir="DATA/asvspoof2019/")
    print("second time: ", time.time() - start_time)
    # print(score_tensor)