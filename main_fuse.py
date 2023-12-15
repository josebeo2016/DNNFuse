import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from model.category import Model as category
from model.scoring import Model as scoring
from model.scoring_joint import Model as scoring_joint
from model.scoring_log import Model as scoring_log
import importlib
import time
from model.loss_metrics import Loss_category_class_CE, Loss_category_CE
from tensorboardX import SummaryWriter

from datautils.score import Score

__author__ = "PhucDT"
__reference__ = "Hemlata Tak"

class EarlyStop:
    def __init__(self, patience=5, delta=0, init_best=60, save_dir=''):
        self.patience = patience
        self.delta = delta
        self.best_score = init_best
        self.counter = 0
        self.early_stop = False
        self.save_dir = save_dir

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("Best epoch: {}".format(epoch))
            self.best_score = score
            self.counter = 0
            # save model here
            torch.save(model.state_dict(), os.path.join(
                self.save_dir, 'epoch_{}.pth'.format(epoch)))
def train_epoch(train_loader, scoring_model, category_model, optimizer, device, loss_func, config):
    scoring_model.train() 
    if config['model']['joint']:
        pass
    else:
        category_model.eval()
    
    running_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    train_loss_detail = {}
    for batch_x, batch_score, batch_class, batch_cate in train_loader:
        train_loss = 0.0
        # print("training on anchor: ", info)
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_score = batch_score.to(device)
        batch_class = batch_class.view(-1).type(torch.int64).to(device)
        batch_cate = batch_cate.view(-1).type(torch.int64).to(device)
        optimizer.zero_grad()

        if config['model']['joint']:
            batch_out, batch_cate_pred = scoring_model(batch_score, batch_x)
        else:
            with torch.no_grad():
                batch_cate_pred, batch_emb = category_model(batch_x)
            batch_out = scoring_model(batch_score, batch_emb)

        if (config['model']['joint']):
            losses = loss_func(batch_cate_pred, batch_cate, batch_out, batch_class)
        else:
            losses = loss_func(batch_out, batch_class)
        
        # train_loss += losses['loss']
        
        for key, value in losses.items():
            train_loss += value
            train_loss_detail[key] = train_loss_detail.get(key, 0) + value.item()
        
        running_loss+=train_loss.item()
        _, batch_pred = batch_out.max(dim=1)
        # batch_y = batch_y.view(-1)
        num_correct += (batch_pred == batch_class).sum(dim=0).item()
        
        
        train_loss.backward()
        optimizer.step()

    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy, train_loss_detail

def evaluate_accuracy(dev_loader, scoring_model, category_model, device, loss_func, config):
    val_loss = 0.0
    val_loss_detail = {}
    num_total = 0.0
    num_correct = 0.0
    if config['model']['joint']:
        scoring_model.eval()
    else:
        scoring_model.eval()
        category_model.eval()
    for batch_x, batch_score, batch_class, batch_cate in dev_loader:
        loss_value = 0.0
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_score = batch_score.to(device)
        batch_class = batch_class.view(-1).type(torch.int64).to(device)
        batch_cate = batch_cate.view(-1).type(torch.int64).to(device)
        if config['model']['joint']:
            batch_out, batch_cate_pred = scoring_model(batch_score, batch_x)
        else:
            with torch.no_grad():
                batch_cate_pred, batch_emb = category_model(batch_x)
            batch_out = scoring_model(batch_score, batch_emb)

        if (config['model']['joint']):
            losses = loss_func(batch_cate_pred, batch_cate, batch_out, batch_class)
        else:
            losses = loss_func(batch_out, batch_class)
            
        for key, value in losses.items():
            loss_value += value
            val_loss_detail[key] = val_loss_detail.get(key, 0) + value.item()
        val_loss+=loss_value.item()
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_class).sum(dim=0).item()
        
    val_loss /= num_total
    val_accuracy = (num_correct/num_total)*100
   
    return val_loss, val_accuracy, val_loss_detail


def produce_evaluation_file(dataset, batch_size, scoring_model, category_model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    num_total = 0.0
    
    if config['model']['joint']:
        scoring_model.eval()
    else:
        scoring_model.eval()
        category_model.eval()
    
    for batch_x, batch_score, utt_id in tqdm(data_loader):
        fname_list = []
        score_list = []
        loss_value = 0.0
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_score = batch_score.to(device)
        
        if config['model']['joint']:
            batch_out, batch_cate_pred = scoring_model(batch_score, batch_x)
        else:
            with torch.no_grad():
                batch_cate_pred, batch_emb = category_model(batch_x)
            batch_out = scoring_model(batch_score, batch_emb)
            
        # batch_out = scoring_model(batch_score, batch_emb)
    
        batch_logit = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        _, batch_pred = batch_out.max(dim=1)

        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_logit.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/', help='eval set')
    '''
    % database_path/
    %   | - protocol.txt
    %   | - audio_path
    '''
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')

    parser.add_argument('--scoring_model', type=str,
                        default=None, help='Scoring model checkpoint')
    parser.add_argument('--score_path', type=str,
                        default=None, help='path to score files of CMS')
    parser.add_argument('--category_model', type=str,
                        default=None, help='Category model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='get the predicted label instead of score')

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('out'):
        os.mkdir('out')
    args = parser.parse_args()
    
    # Score function initialization
    
    
 
        
    # #define model saving path
    model_tag = 'model_{}_{}_{}'.format(
        args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('out', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
    print("Number of GPUs available: ", torch.cuda.device_count())
    
    # load config file
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    # score function
    if (args.score_path):
        score = Score(score_path=args.score_path, CMS=config['model']['cms'])
    else:
        score = Score(score_path=config['data']['score_path'], CMS=config['model']['cms'])
    get_cm_scores = score.get_cm_scores
    
    # dynamic load datautils based on name in config file
    genList = importlib.import_module('datautils.'+ config['data']['name']).genList
    Dataset_train = importlib.import_module('datautils.'+ config['data']['name']).Dataset_train
    Dataset_eval = importlib.import_module('datautils.'+ config['data']['name']).Dataset_eval
    
    # dynamic load model based on name in config file
    category_model = globals()[config['model']['category']](config['model'], device)
    scoring_model = globals()[config['model']['scoring']](config['model'], device)
    
    category_model = category_model.to(device)
    scoring_model = scoring_model.to(device)
    
    # nb_params = sum([param.view(-1).size()[0] for param in category_model.parameters()])
    # print('nb_params:',nb_params)
    
    # dynamic load loss based on name in config file
    loss_function = globals()[config['loss']['name']](config['loss'])

    #set Adam optimizer
    optimizer = torch.optim.Adam(scoring_model.parameters(), lr=args.lr*1000,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr*1000, cycle_momentum=False)
    
    
    if args.scoring_model:
        try:
            scoring_model.load_state_dict(torch.load(args.scoring_model,map_location=device))
        except:
            # fix unexpected key error with module. prefix
            state_dict = torch.load(args.scoring_model,map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            scoring_model.load_state_dict(new_state_dict)
        print('Scoring Model loaded : {}'.format(args.scoring_model))
        
    if args.category_model:
        try:
            category_model.load_state_dict(torch.load(args.category_model,map_location=device))
        except:
            # fix unexpected key error with module. prefix
            state_dict = torch.load(args.category_model,map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            category_model.load_state_dict(new_state_dict)
        print('Category Model loaded : {}'.format(args.category_model))
    
    if torch.cuda.device_count() > 1:
        category_model = nn.DataParallel(category_model)
        scoring_model = nn.DataParallel(scoring_model)
        
    #evaluation 
    if args.eval:
        file_eval = genList(dir_meta = os.path.join(args.database_path,'protocol.txt'),is_train=False,is_dev=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_eval(list_IDs = file_eval, base_dir = os.path.join(args.database_path+'/'), cm_scores=get_cm_scores)
        produce_evaluation_file(eval_set, args.batch_size, scoring_model, category_model, device, args.eval_output)
        sys.exit(0)
   
    # define train dataloader
    d_label_trn, d_cate_trn, file_train = genList(dir_meta = os.path.join(args.database_path,'protocol.txt'),is_train=True,is_eval=False,is_dev=False)
    
    print('no. of training trials',len(file_train))
    
    train_set = Dataset_train(args, list_IDs = file_train, labels = d_label_trn, categories = d_cate_trn,
        base_dir = args.database_path+'/',algo=args.algo, cm_scores=get_cm_scores)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev, d_cate_dev, file_dev = genList(dir_meta = os.path.join(args.database_path,'protocol.txt'),is_train=False,is_eval=False, is_dev=True)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_train(args,list_IDs = file_dev, labels = d_label_dev, categories = d_cate_dev,
		base_dir = args.database_path+'/',algo=args.algo, cm_scores=get_cm_scores)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    
    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    early_stopping = EarlyStop(patience=10, delta=0.0001, init_best=0.04, save_dir=model_save_path)
    start_train_time = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}. Current LR: {}'.format(epoch, num_epochs - 1, optimizer.param_groups[0]['lr']))
        
        running_loss, train_accuracy, train_loss_detail = train_epoch(train_loader,scoring_model, category_model, optimizer, device, loss_function, config)
        val_loss, val_accuracy, val_loss_detail = evaluate_accuracy(dev_loader, scoring_model, category_model, device, loss_function, config)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        for loss_name, loss in train_loss_detail.items():
            writer.add_scalar('train_{}'.format(loss_name), loss, epoch)
        for loss_name, loss in val_loss_detail.items():
            writer.add_scalar('val_{}'.format(loss_name), loss, epoch)
        print('\n{} - {} - {} '.format(epoch,running_loss,val_loss))
        scheduler.step()
        # check early stopping
        early_stopping(val_loss, scoring_model, epoch)
        if early_stopping.early_stop:
            print("Early stopping activated.")
            break
        
    print("Total training time: {}s".format(time.time() - start_train_time))
