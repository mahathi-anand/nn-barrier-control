#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:18:32 2022

@author: mahathi
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import superp_init as superp # parameters
import loss # computing loss
import opt
import lrate


#################################################
# iterative training: the most important function
# it relies on three assistant functions:
#################################################


# used to output learned model parameters
def print_nn(model):
    for p in model.parameters():
        print(p.data)

def print_nn_matlab(model):
    layer = 0
    for p in model.parameters():
        layer = layer + 1
        arr = p.detach().numpy()
        if arr.ndim == 2:
            print( "w" + str((layer + 1) // 2) + " = [", end="")
            print('; '.join([', '.join(str(curr_int) for curr_int in curr_arr) for curr_arr in arr]), end="];\n")
        elif arr.ndim == 1:
            print( "b" + str(layer // 2) + " = [", end="")
            if layer == 2:
                print(', '.join(str(i) for i in arr), end="]';\n")
            else:
                print(', '.join(str(i) for i in arr), end="];\n")
        else:
            print("Transform error!")

# used for initialization and restart
def initialize_nn(model, stored_name, num_batches):    
    print("Initialize nn parameters!")

    ## random initialize or load saved
    if superp.FINE_TUNE == 0:
        for p in model.parameters():
            nn.init.normal_(p,0,0.01) #standard Gaussian distribution
    else:
        for p in model.parameters():
            p.requires_grad=True

    ## fix parameters
    # if fixed == True:
    #     for p in model.parameters():
    #         p.requires_grad = False

    optimizer = opt.set_optimizer(model)
    scheduler = lrate.set_scheduler(optimizer, num_batches)

    return optimizer, scheduler

def initialize_parameters(n_h_b, d_h_b, n_h_c, d_h_c):
    #initialize the eta variable for scenario verification
    
    print("Initialize eta")
    
    eta=Variable(torch.normal(mean=torch.tensor([-0.0035]), std=torch.tensor([0.0001])), requires_grad=True)
    
    #initialize the lambda matrix for Lipschitz constant computation
    
    # print("Initialize lambda")
    
    lambdas_b=Variable(torch.normal(mean=10*torch.ones(n_h_b*d_h_b),std=0.001*torch.ones(n_h_b*d_h_b)), requires_grad=True)
    
    lambdas_c=Variable(torch.normal(mean=10*torch.ones(n_h_c*d_h_c),std=0.001*torch.ones(n_h_c*d_h_c)), requires_grad=True)

    
    return eta,lambdas_b, lambdas_c
    
# to prevent generating a nn with large gradient, it works only for nn model with a single hidden layer
# do we need scale_ctrl? regularization？
# def scale_nn(model, scale_factor): 
#     with torch.no_grad():
#         print("Scale nn parameters!")
#         i = 0
#         for p in model.parameters(): # i = 1, 3, 5, 7: weight matrix; i = 2, 4, 6, 8: bias
#             i = i + 1
#             if i % 2 == 0:
#                 p.data = p.data / torch.pow(torch.tensor(scale_factor), i // 2)
#             else:
#                 p.data = p.data / scale_factor

def itr_train(barr_nn, ctrl_nn, batches_init, batches_unsafe, batches_domain, NUM_BATCHES):
    # set the number of restart times
    
    num_restart = -1

    ############################## the main training loop ##################################################################
    while num_restart < 4:
        num_restart += 1
        
        # initialize nn models and optimizers and schedulers
        
        optimizer_barr, scheduler_barr = initialize_nn(barr_nn, "barr_nn", NUM_BATCHES[3])
        optimizer_ctrl, scheduler_ctrl = initialize_nn(ctrl_nn, "ctrl_nn", NUM_BATCHES[3])
      #  eta,
        eta, lambdas_b, lambdas_c= initialize_parameters(superp.N_H_B, superp.D_H_B, superp.N_H_C, superp.D_H_C)

        # check counter example by isat3
        # loss.test_lie(barr_nn, ctrl_nn, [-0.316, 0.0477, 0.0367])

        init_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[0]  # generate batch indices    # I
        unsafe_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[1]                            # U
        domain_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[2]                            # D
#        asymp_list = np.arange(NUM_BATCHES[4]) % NUM_BATCHES[3]                             # A

        for epoch in range(superp.EPOCHS): # train for a number of epochs
            # initialize epoch
            epoch_loss = 0 # scalar
            lmi_loss = 0 #scalar
            eta_loss = 0
            epoch_gradient_flag = True # gradient is within range
            superp.CURR_MAX_GRAD = 0

            # mini-batches shuffle by shuffling batch indices
            np.random.shuffle(init_list) 
            np.random.shuffle(unsafe_list)
            np.random.shuffle(domain_list)

            # train mini-batches
            for batch_index in range(NUM_BATCHES[3]):
                # batch data selection
                batch_init = batches_init[init_list[batch_index]]
                batch_unsafe = batches_unsafe[unsafe_list[batch_index]]
                batch_domain = batches_domain[domain_list[batch_index]]

                ############################## mini-batch training ################################################
                optimizer_barr.zero_grad() # clear gradient of parameters
                optimizer_ctrl.zero_grad()
                #eta=torch.tensor(-0.003)
                curr_batch_loss = loss.calc_loss(barr_nn, ctrl_nn, batch_init, batch_unsafe, batch_domain, epoch, batch_index,eta, superp.lip_b, superp.lip_c)
                    # batch_loss is a tensor, batch_gradient is a scalar
                if curr_batch_loss.item() > 0:
                    curr_batch_loss.backward() # compute gradient using backward()
                    # update weight and bias
                    optimizer_barr.step() # gradient descent once
                    optimizer_ctrl.step()
                   
                optimizer_barr.zero_grad()
                
                curr_lmi_loss= loss.calc_lmi_loss(barr_nn, ctrl_nn, lambdas_b, lambdas_c, superp.lip_b, superp.lip_c)
                                
                if curr_lmi_loss >= -50:
                    curr_lmi_loss.backward()
                    optimizer_barr.step()
                    optimizer_ctrl.step()
                    
                optimizer_barr.zero_grad()
                
                curr_eta_loss= loss.calc_eta_loss(eta, superp.lip_b, superp.lip_c)
                
                if curr_eta_loss > 0:
                    curr_eta_loss.backward()
                    optimizer_barr.step()
                    optimizer_ctrl.step()

                # learning rate scheduling for each mini batch
                scheduler_barr.step() # re-schedule learning rate once
                scheduler_ctrl.step()


                # update epoch loss
                epoch_loss += curr_batch_loss.item()
                lmi_loss += curr_lmi_loss
                eta_loss += curr_eta_loss
                
                # update epoch gradient flag
#                curr_batch_gradient_flag = curr_batch_gradient < superp.TOL_MAX_GRAD
#                epoch_gradient_flag = epoch_gradient_flag and curr_batch_gradient_flag

          #      if curr_batch_gradient > superp.CURR_MAX_GRAD:
           #         superp.CURR_MAX_GRAD = curr_batch_gradient

                if superp.VERBOSE == 1:
                    print("restart: %-2s" % num_restart, "epoch: %-3s" % epoch, "batch: %-5s" % batch_index, "batch_loss: %-25s" % curr_batch_loss.item(), \
                          "epoch_loss: %-25s" % epoch_loss, "lmi loss: % 25s" %lmi_loss, "eta loss: % 25s" %eta_loss, "eta:" % eta)

                # gradient control by scale parameters  #What is this gradient scaling?
                # if not curr_batch_gradient_flag:
                #     scale_nn(barr_nn, superp.GRAD_CTRL_FACTOR)

            if (epoch_loss < superp.LOSS_OPT_FLAG) and (lmi_loss <= 0) and (eta_loss < superp.LOSS_OPT_FLAG):
                print("The last epoch:", epoch, "of restart:", num_restart)
                if superp.VERBOSE == 1:
                    print("\nSuccess! The nn barrier is:")
                    print_nn_matlab(barr_nn) # output the learned model
                    print("\nThe nn controller is:")
                    print_nn_matlab(ctrl_nn)
                    print("\nThe value of eta is:")
                    print(eta)

                return True # epoch success: end of epoch training

    return False
