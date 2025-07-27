#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:53:30 2022

@author: mahathi
"""
import torch
import torch.nn as nn
import numpy as np


############################################
## This code is for initializing the system dimension
## and training (NN) parameters
###########################################

############################################
# set default data type to double; for GPU
# training use float
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)

VERBOSE = 1 # set to 1 to display epoch and batch losses in the training process
VISUAL = 0 # plot figure or not

FINE_TUNE = 0 # set to 1 for fine-tuning a pre-trained model
FIX_CTRL = 0
FIX_BARR = 0


############################################
# set the system dimension
############################################
DIM_S = 2 # dimension of system
DIM_C = 2 # dimension of controller input

############################################
# set the network architecture
############################################
N_H_B = 1 # the number of hidden layers for the barrier
D_H_B = 20 # the number of neurons of each hidden layer for the barrier

N_H_C = 1 # the number of hidden layers for the controller
D_H_C = 5 # the number of neurons of each hidden layer for the controller

############################################
# for activation function definition
############################################
#BENT_DEG = 0.0001

#BARR_ACT = acti.my_act(BENT_DEG)  #Bent_relu is needed for differentiability. Since we
# deal with discrete-time systems, no differentiability is needed.

BARR_ACT=nn.ReLU() #use relu for both barrier and controller
CTRL_ACT = nn.ReLU()

BARR_IN_BOUND = -1e16
BARR_OUT_BOUND = 1e16 # set the output bound of the barrier NN

#Change this if you want the controller to be bounded! 
CTRL_IN_BOUND= -10
CTRL_OUT_BOUND = 10 # set the output bound of the controller NN: for bounded controller

###########################################
#Barrier certificate conditions
#########################################

gamma=0; #first condition <= 0
lamda=0.001; #this is required for strict inequality >= lambda

#eta=-0.05 #fix the eta for the SCP problem

############################################
# set loss function definition
############################################
TOL_INIT = 0.0   # tolerance for initial and unsafe conditions
TOL_UNSAFE = 0.000

TOL_LIE = 0.000 #tolerance for the last condition

TOL_DATA_GEN = 1e-16 #for data generation


############################################
#Lipschitz bound for training
lip_b=2
lip_c=20
############################################
# number of training epochs
############################################
EPOCHS = 500

############################################
# my own scheduling policy: 
# rate = alpha / (1 + beta * epoch^gamma)
############################################
#ALPHA = 0.035 # initial learning rate
ALPHA=0.05
BETA = 0 # if beta equals 0 then constant rate = alpha
GAMMA = 0 # when beta is nonzero, larger gamma gives faster drop of rate

#weights for loss function

DECAY_LIE = 1 # decay of lie weight 0.1 works, 1 does not work
DECAY_INIT = 1
DECAY_UNSAFE = 1

############################################
# training termination flags

LOSS_OPT_FLAG = 1e-16
TOL_MAX_GRAD = 5
GRAD_CTRL_FACTOR = 1.4

