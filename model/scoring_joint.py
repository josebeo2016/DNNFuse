import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .linear import MLP
from .category import Model as category_model

class Model(nn.Module):
    def __init__(self, config, device):
        """
        config: dict, configuration for the model
        device: torch.device, device to run the model
        """
        super(Model, self).__init__()
        self.device=device
        self.is_train = True
        self.category_model = category_model(config, device)
        self.weight_prj = MLP(config['out_dim'], config['out_dim'], 1, config['num_cms'], config['dropout_rate'])
        self.is_norm = config['is_norm']

    def forward(self, scores, x):
        """
        scores: tensor, (batch, 2, #number CM): the scores for 2 classes from #number CM
        x: tensor, (batch, audio_len) for raw audio input
        """
        # category model
        cate_out, emb = self.category_model(x)
        if self.is_norm:
            # normalize the scores
            scores = F.softmax(scores, dim=1)
            # print("score:", scores)
            # project the embedding to the scores
            emb, _ = self.weight_prj(emb) #(bs, out_dim) -> (bs, #number CM)
            # normalize the embedding
            emb = F.softmax(emb, dim=1)
            # print("emb", emb.shape)
            emb = emb.unsqueeze(2) #(bs, #number CM, 1)
            # print("emb", emb)
            # weighted sum of the scores
            out = torch.matmul(scores, emb) #(bs, 2, #number CM) * (bs, #number CM, 1) = (bs, 2, 1)
            out = out.squeeze(2) #(bs, 2)
            # out = torch.log(out)
            # print(out)

        else:
            # project the embedding to the scores
            emb, _ = self.weight_prj(emb) #(bs, out_dim) -> (bs, #number CM)
            emb = F.softmax(emb, dim=1)
            emb = emb.unsqueeze(2) #(bs, #number CM, 1)
            # weighted sum of the scores
            out = torch.matmul(scores, emb) #(bs, 2, #number CM) * (bs, #number CM, 1) = (bs, 2, 1)
            out = out.squeeze(2) #(bs, 2)
            # out = torch.log(out)
            # print(out)
        return out, cate_out