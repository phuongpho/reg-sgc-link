import os
import torch
import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score


def rand_link_split(u, v, train_val_test, pos_size = None, seed = None):
    '''
    This function randomly splits existing links into training, validating, and test samples.
    Args:
        u (tensor): list of source node ids
        v (tensor): list of destination node ids
        train_val_test: list of proportion of train/val/test
        pos_size (optional): number of positive samples
        seed: random seed
    '''
    # Remove reverse edges for undirected graph
    mask = u <= v
    eids = mask.nonzero(as_tuple=False).view(-1)
    
    # Random sampling link
    if seed is not None:
        torch.manual_seed(seed)
          
    if pos_size is None:
        eids = np.random.permutation(eids)
    else:
        eids = eids[np.random.choice(len(eids), pos_size)]
        eids = np.unique(eids)

    _,val_prop,test_prop = train_val_test
    
    # Commpute sizes of train, val, test sets
    test_size = int(len(eids) * test_prop)
    val_size = int(len(eids) * val_prop)
    train_size = len(eids) - (val_size + test_size)
    train_val_size = train_size + val_size
    
    print(f'Total num edges: {len(eids)}, Train edges: {train_size}, val edges: {val_size}, test edges: {test_size}')
    # Split
    train_u, train_v = u[eids[:train_size]], v[eids[:train_size]]
    val_u, val_v = u[eids[train_size : train_val_size]], v[eids[train_size : train_val_size]]
    test_u, test_v = u[eids[train_val_size:]], v[eids[train_val_size:]]
    
    return (train_u, train_v, val_u, val_v, test_u, test_v)

def balance_sample(pos_sample, neg_sample):
    '''
    This function adjusts the balance between pos samples and neg samples.
    In sparse graph, negative samples might greately exceed positive samples. 
    Args:
        pos_sample (tensor): list of positive ids
        neg_sample (tensor): list of negative ids
    '''
    if len(pos_sample) < len(neg_sample):
        neg_sample = neg_sample[:len(pos_sample)]
    return (pos_sample, neg_sample)

def data_loader(args, train_val_test = [0.7,0.1,0.2]):
    
    dataset = args.dataset
    
    if hasattr(args,'seed'):
        seed = args.seed
    else:
        seed = None
    
    print(f'Loading {dataset} dataset')
    # load and preprocess dataset
    if dataset == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    else:
        ''' For custome datasets stored in data/.'''
        dt_name = os.listdir('./data')
        check_name = []
        
        for i,v in enumerate(dt_name):
            if args.dataset+'.dgl' == v:
                check_name.append(True)
                check_id = i
            else:
                check_name.append(False)
        
        if any(check_name):
            fl_path = './data/'+dt_name[check_id]
            data,_ = dgl.load_graphs(fl_path)
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # extract graph data
    g = data[0]

    if args.gpu < 0:
        device = 'cpu'
    else:
        device = 'cuda'
    
    g.to(device)

    # Train test split
    u,v = g.edges()
    eids = np.arange(g.number_of_edges())

    # Generate pos links
    print('Generating pos links')
    train_pos_u, train_pos_v, val_pos_u, val_pos_v, test_pos_u, test_pos_v = rand_link_split(u, v, train_val_test, seed = seed)

    # Find all negative edges 
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_u = torch.from_numpy(neg_u)
    neg_v = torch.from_numpy(neg_v)
    
    print('Generating negative link')
    train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_u, test_neg_v = rand_link_split(neg_u, neg_v, train_val_test, pos_size = g.number_of_edges()//2)

    # Checking balance between pos and neg samples
    if len(train_pos_u) < len(train_neg_u):
        print(f'Train pos sample: {len(train_pos_u)} < Train neg sample: {len(train_neg_u)}. Adjusting train neg sample')
        train_pos_u, train_neg_u = balance_sample(train_pos_u, train_neg_u)
        train_pos_v, train_neg_v = balance_sample(train_pos_v, train_neg_v)

    if len(val_pos_u) < len(val_neg_u):
        print(f'Val pos sample: {len(val_pos_u)} < Val neg sample: {len(val_neg_u)}. Adjusting val neg sample')
        val_pos_u, val_neg_u = balance_sample(val_pos_u, val_neg_u)
        val_pos_v, val_neg_v = balance_sample(val_pos_v, val_neg_v)

    if len(test_pos_u) < len(test_neg_u):
        print(f'Test pos sample: {len(test_pos_u)} < Test neg sample: {len(test_neg_u)}. Adjusting test neg sample')
        test_pos_u, test_neg_u = balance_sample(test_pos_u, test_neg_u)
        test_pos_v, test_neg_v = balance_sample(test_pos_v, test_neg_v)

    # remove val and test edges in original graph for training purpose
    # by remove all edges from g and add train edge, then convert to bidirected
    train_g = dgl.remove_edges(g, eids)

    # print(f'Is train_g multigraph:{train_g.is_multigraph}')
    # Construc pos, neg graphs
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes = g.number_of_nodes())
    train_pos_g = dgl.to_bidirected(train_pos_g)

    # assign train_g as train_pos_g
    train_g = train_pos_g

    # print(f'Is train_pos_g multigraph:{train_pos_g.is_multigraph}')
    
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes = g.number_of_nodes())
    
    # Check and remove duplicated links in graph
    if train_neg_g.is_multigraph:
        print('Duplicated link in train_neg_g')
        train_neg_g = dgl.to_simple(train_neg_g)
        print(f'New train neg size: {train_neg_g.number_of_edges()}')
    train_neg_g = dgl.to_bidirected(train_neg_g)

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.to_bidirected(test_pos_g)

    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    if test_neg_g.is_multigraph:
        print('Duplicated link in test_neg_g')
        test_neg_g = dgl.to_simple(test_neg_g)
        print(f'New test neg size: {test_neg_g.number_of_edges()}')
    test_neg_g = dgl.to_bidirected(test_neg_g)

    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.number_of_nodes())
    val_pos_g = dgl.to_bidirected(val_pos_g)

    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.number_of_nodes())
    if val_neg_g.is_multigraph:
        print('Duplicated link in val_neg_g')
        val_neg_g = dgl.to_simple(val_neg_g) 
        print(f'New val neg size: {test_neg_g.val_neg_g()}')

    val_neg_g = dgl.to_bidirected(val_neg_g)
    
    return (g, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, val_pos_g, val_neg_g)


# Make trainer function to perform training
def make_trainer(model, pred, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def trainer(features, g, pos_g, neg_g):
        # Set model and pred to TRAIN mode
        model.train()
        pred.train()

        L1 = model.L1
        L2 = model.L2
        
        # Makes predictions
        h = model(g, features)
        
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
        
        # Compute the loss
        if hasattr(pred, 'W1'):
            loss = loss_fn(pos_score, neg_score, pred.W1.weight, L1, L2)
        else:
            loss = loss_fn(pos_score, neg_score)
       
        # Computes gradients
        loss.backward()
        
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return trainer

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def evaluate(model, pred, loss_fn, train_g, features, eval_pos_g, eval_neg_g):
    model.eval()
    pred.eval()
    with torch.no_grad():
        L1 = model.L1
        L2 = model.L2
        
        h = model(train_g, features)

        pos_score = pred(eval_pos_g, h)
        neg_score = pred(eval_neg_g, h)

        # Compute the loss
        if hasattr(pred, 'W1'):
            loss = loss_fn(pos_score, neg_score, pred.W1.weight, L1, L2)
        else:
            loss = loss_fn(pos_score, neg_score)
        
        # Compute AUC
        auc = compute_auc(pos_score, neg_score)
        
        return loss.item(), auc

class EarlyStopping:
    def __init__(self, path, patience=10, verbose=False, delta = 0.0, **metric_direction):
        '''
        Args:
            path (str): Path to save model's checkpoint.
            patience (int): How long to wait after last time validation loss (or acc) improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss (or acc) improvement. 
                            Default: False
            delta (float): Minimum percentage change in the monitored quantity (either validation loss or acc) to qualify as an improvement.
                            Default: 0.0
            **metric_direction: Keywords are names of metrics used for early stopping. Values are direction in ['low'/'high']. Use 'low' if a small quantity of metric,
                            is desirable for training and vice versa. E.g: loss = 'low', acc = 'high'. If not provided, use loss = 'low'
        '''
        if metric_direction:
            print(f'Selected metric for early stopping: {metric_direction}')
        else:
            raise ValueError("No metric provided for early stopping")

        # unpacking keys into list of string
        self.metric_name = [*metric_direction.keys()]
        # choose comparison operator w.r.t metric direction: low -> "<"; high -> ">"
        self.metric_operator = [np.less if dir == 'low' else np.greater for dir in metric_direction.values()]
        self.patience = patience
        # assign delta sign to compute reference quantity for early stopping
        self.delta = [-delta if dir == 'low' else delta for dir in metric_direction.values()]
        self.counter = 0
        self.best_score = [None]*len(metric_direction.keys())
        self.best_epoch = None
        self.lowest_loss = None
        self.path = path
        self.verbose = verbose
        self.early_stop = False
          
    def __call__(self, model, pred, epoch, args, **metric_value):
        '''
        Args:
            metric_value: Keywords are names of metrics used for early stopping. Values are metrics's value obtained during training.
        '''
        if metric_value:
            # Check name of metric
            if set(self.metric_name) != set(metric_value.keys()):
                raise ValueError("Metric name is not matching")
        else:
            raise ValueError("Metric value is missing")
        
        score = [metric_value[key] for key in self.metric_name if key in metric_value]
        
        # if any metric is none, return true
        is_none = any(map(lambda i: i is None,self.best_score))
        
        if is_none:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model, pred, args)
        else:
            # score condition: if any metric is getting better, save model. 
            # getting better means scr is less(greater) than best_scr*[1-(+)delta/100]
            score_check = any(map(lambda scr,best_scr, op, dlt: op(scr, best_scr*(1+dlt/100)), score, self.best_score, self.metric_operator, self.delta))
            
            if score_check:
                self.best_score = score
                self.best_epoch = epoch
                self.save_checkpoint(model, pred, args)
            else:
                self.counter += 1
                if self.counter >= 0.8*(self.patience):
                    print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            return self.early_stop

    def save_checkpoint(self, model, pred, args):
        '''
        Saves model when score condition is met, i.e loss decreases 
        or acc increases
        '''
        if self.verbose:
            message = f'Model saved at epoch {self.best_epoch + 1}.'
            score = self.best_score
            
            if len(self.metric_name) > 1:
                for i,nm in enumerate(self.metric_name):
                    message += f' {nm}={score[i]:.4f}'
                
                print(message)
            else:
                print(f'{message} {self.metric_name[0]}={score[0]:.4f}')
        # Save model state
        torch.save({
            'model_state_dict':model.state_dict(),
            'pred_state_dict':pred.state_dict(),
            'args': vars(args)
        }, self.path)

# Function to import data to dgl graph object
def import_data(filename, sp_Adj ,X,labels, train_mask, val_mask, test_mask, pred_mask = None):
    '''
    This function converts inputs data into dgl graph object.
    Args:
        filename (str): file name without extension
        sp_Adj (scipy.sparse.coo.coo_matrix): adjancency matrix in coo_matrix format
        X (numpy.ndarray): 2D numpy array of features matrix
        labels (numpy.ndarray): 1D numpy array of node labels 
        train_mask (numpy.ndarray): 1D numpy array of mask of training nodes
        val_mask (numpy.ndarray): 1D numpy array of mask of validating nodes
        test_mask (numpy.ndarray): 1D numpy array of mask of test nodes
        pred_mask (numpy.ndarray): 1D numpy array of mask of unlabeled nodes for prediction
    '''
    fl_path = './data/' + filename + '.dgl'
    # Convert to dgl graph 
    G = dgl.from_scipy(sp_Adj)
    
    # Add feature
    G.ndata['feat'] = torch.FloatTensor(X)

    # Add labels
    G.ndata['label'] = torch.tensor(labels, dtype=torch.long)

    # Add train-val-test-pred masks
    G.ndata['train_mask'] = torch.tensor(train_mask, dtype= torch.bool)
    G.ndata['test_mask'] = torch.tensor(test_mask, dtype= torch.bool)
    G.ndata['val_mask'] = torch.tensor(val_mask, dtype= torch.bool)
    
    if pred_mask is not None:
        G.ndata['pred_mask'] = torch.tensor(pred_mask, dtype= torch.bool)
    
    # Save graph
    dgl.save_graphs(fl_path,G)

