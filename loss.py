#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:27:15 2022

@author: mahathi
"""

import torch
import torch.nn as nn
import superp_init as superp
import prob
import torch.nn.functional as F
import data
############################################
# constraints for barrier certificate B:
# (1) init ==> B <= 0
# (2) unsafe ==> B >= lambda 
# where lambda>0
# (3) domain ==> B(f(x))-B <= 0
############################################


############################################
# given the training data, compute the loss
############################################

def lipschitz(lambdas, lip, model):
    
    weights=[];
    layer=0;
    for p in model.parameters():
        if layer % 2 == 0:
            weights.append(p.data)
        layer = layer+1
    
    T= torch.diag(lambdas)
    
    diag_items= [lip**2*torch.eye(superp.DIM_S),  2*T,  torch.eye(1)]
    
    subdiag_items= [torch.matmul(T, weights[0]), weights[-1]]
    
    dpart = torch.block_diag(diag_items[0],diag_items[1],diag_items[2])
    
    spart= F.pad(torch.block_diag(subdiag_items[0],subdiag_items[1]), (0,1, superp.DIM_S, 0))
    
    return dpart-spart-torch.transpose(spart,0,1)
    
    
def calc_loss(barr_nn, ctrl_nn, input_init, input_unsafe, input_domain, epoch, batch_index, eta,lip_b,lip_c):
    # compute loss of init    
    output_init = barr_nn(input_init)
    loss_init = torch.relu(output_init - superp.gamma + superp.TOL_INIT - eta) #tolerance

    # compute loss of unsafe
    output_unsafe = barr_nn(input_unsafe)
    loss_unsafe = torch.relu((- output_unsafe) + superp.lamda + superp.TOL_UNSAFE - eta) #tolerance
    
    # compute loss of domain
    output_domain=barr_nn(input_domain)
    
    vector_domain = prob.vector_field(input_domain, ctrl_nn) # compute vector field at domain
    output_vector=barr_nn(vector_domain)
    loss_lie=torch.relu(output_vector-output_domain + superp.TOL_LIE-eta)
    
    #loss_eta=torch.relu(torch.tensor(lip_b*(prob.L_x + prob.L_c* lip_c)) + eta)
        
    total_loss = superp.DECAY_INIT * torch.sum(loss_init) + superp.DECAY_UNSAFE * torch.sum(loss_unsafe) \
                    + superp.DECAY_LIE * torch.sum(loss_lie) #+ loss_eta
                    
    # return total_loss is a tensor, max_gradient is a scalar
    return total_loss

def calc_lmi_loss(barr_nn,ctrl_nn,lambdas_b, lambdas_c, lip_b, lip_c):
    
      lmi_loss = -0.001*(torch.logdet(lipschitz(lambdas_b, lip_b, barr_nn)) + torch.logdet(lipschitz(lambdas_c, lip_c, ctrl_nn)))
    
      return lmi_loss

def calc_eta_loss(eta, lip_b, lip_c):
    
     loss_eta=torch.relu(torch.tensor(lip_b*(prob.L_x + prob.L_u* lip_c)*data.eps+lip_b*data.eps) + eta)
    
     return loss_eta

    
    

    
    
