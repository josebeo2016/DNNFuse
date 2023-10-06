import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_category_class_CE(nn.Module):
    def __init__(self, config):
        """
        config: dict, configuration for the loss
        """
        super(Loss_category_class_CE, self).__init__()
        self.loss_ce1 = nn.CrossEntropyLoss()
        self.loss_ce2 = nn.CrossEntropyLoss()
        self.config = config
    
    def forward(self, y_cate_hat, y_cate, y_class_hat, y_class):
        """
        y_cate_hat: tensor, (batch, num_classes), predicted category
        y_cate: tensor, (batch), ground truth category
        y_class_hat: tensor, (batch, out_dim), predicted class
        y_class: tensor, (batch), ground truth class
        """
        # category loss
        loss_cate = self.loss_ce1(y_cate_hat, y_cate)
        # class loss
        loss_class = self.loss_ce2(y_class_hat, y_class)
        # # total loss
        # loss = self.config['lambda_cate'] * loss_cate + self.config['lambda_class'] * loss_class
        return {
            'loss_category': loss_cate * self.config['lambda_cate'], 
            'loss_class': loss_class * self.config['lambda_class'],
                # 'loss': loss_cate * self.config['lambda_cate'] + loss_class * self.config['lambda_class']
            }
        
        # return loss, loss_cate, loss_class
        

class Loss_category_CE(nn.Module):
    def __init__(self, config):
        """
        config: dict, configuration for the loss
        """
        super(Loss_category_CE, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.config = config
    
    def forward(self, y_cate_hat, y_cate):
        """
        y_cate_hat: tensor, (batch, num_classes), predicted category
        y_cate: tensor, (batch), ground truth category
        """
        # category loss
        loss_cate = self.loss(y_cate_hat, y_cate)

        return {'loss_category': loss_cate}
        
        # return loss, loss_cate, loss_class
        
class Loss_class_CE(nn.Module):
    def __init__(self, config):
        """
        config: dict, configuration for the loss
        """
        super(Loss_category_CE, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.config = config
    
    def forward(self, y_class_hat, y_class):
        """
        y_class_hat: tensor, (batch, num_classes), predicted category
        y_class: tensor, (batch), ground truth category
        """
        # category loss
        loss_class = self.loss(y_class_hat, y_class)

        return {'loss_class': loss_class}
        
        # return loss, loss_cate, loss_class