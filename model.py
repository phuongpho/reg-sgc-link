from dgl.nn.pytorch.conv import SGConv
import dgl.function as fn
import torch.nn as nn

class regSGConv(SGConv):
    def __init__(self,
                in_feats,
                out_feats,
                L1 = 0.0,
                L2 = 0.0,
                k = 1,
                cached=False,
                bias=True,
                norm=None,
                allow_zero_in_degree=False):
        super().__init__(
                in_feats,
                out_feats,
                k=k,
                cached=cached,
                bias=bias,
                norm=norm,
                allow_zero_in_degree=allow_zero_in_degree)
        
        # Check regularization mode
        reg_zero = [L1 == 0.0, L2 == 0.0]
        if all(reg_zero):
                print("Initializing SGC without regularization !!!")
        
        self.L1 = float(L1)
        self.L2 = float(L2)
        
        print(f"Initializing regularized SGC: L1 = {self.L1}, L2 = {self.L2}")



# Link predictor using Hadamard product combined with SLP
class SLPLinkPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats, 1)
        print('Use Hadamard product to compute edge embedding')

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            #print(f'h device in SLP is: {h.device}, g device is: {g.device}, W1: {next(self.parameters()).device}')
            g.apply_edges(fn.u_mul_v('h', 'h', 'score'))
            return self.W1(g.edata['score']).squeeze(1)

class DotPredictor(nn.Module):
    print('Use dot product as link predictor')
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]