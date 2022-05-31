import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import dgl.function as fn
from dgl.data import register_data_args
from model import *
from loss import *
from utils import *

def main(args):
    # Load dataset  
    g, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, val_pos_g, val_neg_g = data_loader(args)
   
    features = g.ndata['feat']

    in_feats = features.shape[1]
    
    # Add self loop
    train_g = dgl.remove_self_loop(train_g)
    train_g = dgl.add_self_loop(train_g)
    
    # Initialize SGC and link predictor
    model = regSGConv(in_feats, 
                      args.num_hidden,
                      L1 = args.L1,
                      L2 = args.L2,
                      k = args.k,
                      cached = True,
                      bias = args.bias)

    # Assign link predictor
    pred = SLPLinkPredictor(args.num_hidden)
    
    # Checkpoint path  
    checkpoints_path = f'./checkpoints/{args.dataset}_sgc+k_{args.k}+L1_{args.L1}+L2_{args.L2}.pt'
    
    if args.gpu < 0:
        device = 'cpu'
    else:
        device = 'cuda'

    model.to(device)
    pred.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()),
                                 lr=args.lr)
    # use early stop
    if args.early_stop:
        metric_direction = dict(zip(args.early_stop_metric.split(','),args.early_stop_metric_dir.split(',')))
        stopper = EarlyStopping(checkpoints_path, 
                                patience=args.early_stop_patience, 
                                verbose= args.early_stop_verbose, 
                                delta = args.early_stop_delta, 
                                **metric_direction)

    # Create trainer
    train_model = make_trainer(model, pred, reg_loss_lk, optimizer)
    
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()

        loss = train_model(features, train_g, train_pos_g, train_neg_g)

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_loss, val_auc = evaluate(model, pred, reg_loss_lk, train_g, features, val_pos_g, val_neg_g)
        
        if (epoch + 1) % args.hist_print == 0:
                print("Epoch {:05d} | Time(s) {:.4f} | Training Loss {:.4f} | Validation Loss {:.4f} | Validation AUC {:.4f} | "
                        .format(epoch + 1, np.mean(dur), loss, val_loss,
                                                val_auc))
        if args.early_stop:
                val_dict = dict(zip(['loss','auc'],[val_loss,val_auc]))
                metric_value = dict([(key, val_dict[key]) for key in metric_direction.keys() if key in val_dict])
                
                if stopper(model, pred, epoch, args, **metric_value):
                        print(f'Best model achieved at epoch: {stopper.best_epoch + 1}')
                        break
   
    print()
    if args.save_trained:
        print('Saving trained model at ./checkpoints')
        torch.save({
            'model_state_dict':model.state_dict(),
            'pred_state_dict':pred.state_dict(),
            'args': vars(args)
        }, checkpoints_path)
    
    if args.early_stop:
        print('loading model before testing.')
        framework_checkpoint = torch.load(checkpoints_path,
                                      map_location=lambda storage, loc: storage)
        model.load_state_dict(framework_checkpoint['model_state_dict'])
        pred.load_state_dict(framework_checkpoint['pred_state_dict'])
    
    _,auc = evaluate(model, pred, reg_loss_lk, train_g, features, test_pos_g, test_neg_g)
    print("Test AUC {:.4f}".format(auc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regularized SGC')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.2,
            help="learning rate")
    parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
    parser.add_argument("--num-hidden", type=int, default=32,
            help="number of hidden unit for SGC")
    parser.add_argument("--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--k", type=int, default=2,
            help="number of hops")
    parser.add_argument("--L1", type=float, default=0.0,
            help="L1 constraint")
    parser.add_argument("--L2", type=float, default=0.0,
            help="L2 constraint")
    parser.add_argument("--hist-print", type = int, default=10,
            help="print training history every t epoch (default value is 10)")
    parser.add_argument("--save-trained", action='store_true', default = False,
            help="flag to save trained model")
    parser.add_argument("--early-stop", action = 'store_true', default=False,
            help="flag for early stopping")
    parser.add_argument("--early-stop-patience", type = int, default=10,
            help="patience setting for early stopping. Default is 10")
    parser.add_argument("--early-stop-metric", type = str, default = 'loss',
            help="metric used for early stopping. Default is loss")
    parser.add_argument("--early-stop-metric-dir", type = str, default = 'low',
            help="direction of metric used for early stopping [low/high]. Default is low")
    parser.add_argument("--early-stop-delta", type = float, default = 0.0,
            help="delta value used for early stopping. Default is 0.0") 
    parser.add_argument("--early-stop-verbose", action='store_true', default = False,
            help="flag to print message for early stopping")   
    args = parser.parse_args()
    print(args)

    main(args)