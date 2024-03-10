#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def Discrimination(pred, attribute):
    ''' Discrimination for fairness constrain on categorical attribute group'''
    global_mean = torch.mean(pred, axis=0)
    # get unique group from attribute
    att_val = torch.unique(attribute, return_inverse = False, return_counts = False)
    mean_diff = torch.zeros((len(att_val), mean.size()[0]), device = pred.device())
    for i, val in enumerate(att_val): 
        parity_diff[i] = (torch.abs(torch.mean(pred[(attribute == val).nonzero()], axis=0) - mean))
    discrim = 2 * torch.mean(parity_diff, axis=0)
    return discrim

def CCCLoss(x, y, eps = 1e-8):
    ''' Concordance correlation coefficient loss'''
    vx = x - x.mean(dim=0)
    vy = y - y.mean(dim=0)  
    
    pcc = torch.sum(vx * vy, dim=0) / \
            torch.sqrt(torch.add(torch.sum(vx ** 2, dim=0) * \
                torch.sum(vy ** 2, dim=0), eps))
    ccc = (2 * pcc * x.std(dim=0) * y.std(dim=0)) / \
            torch.add(x.var(dim=0) + y.var(dim=0) + ((x.mean(dim=0) - y.mean(dim=0)) ** 2), eps)
    return 1 - ccc
