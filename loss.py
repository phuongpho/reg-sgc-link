import torch
import torch.nn as nn 
import torch.nn.functional as F

def norm_weight(weights):
    # Compute the norm of thetas:
    norm_theta = torch.norm(weights,dim = 1)
    
    # Normalize thetas:
    normalized_theta = nn.functional.normalize(weights, p = 2)

    return normalized_theta, norm_theta

def reg_loss_lk(pos_score, neg_score, weights = None, L1 = 0.0, L2 = 0.0, L3 = 0.0):
    # Concatenate scores and generate corresponding labels
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
      
    # Compute the loss
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    
    if weights is None: 
        return loss
    else:
        # Normalized weights
        normalized_theta, norm_theta = norm_weight(weights)

        # L1 penalty
        if L1 != 0:
            l1_pen = ((normalized_theta**(4)).sum(axis = 1)**(-1)).sum(axis = 0)
        else:
            l1_pen = 0.0
        
        # L2 penalty
        if L2 != 0:
            l2_pen = norm_theta.sum(axis = 0)
        else:
            l2_pen = 0.0
                  
        # Add penalty term to loss
        loss += L1*l1_pen + L2*l2_pen

    return loss
    

