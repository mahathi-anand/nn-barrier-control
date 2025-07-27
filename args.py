#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:40:52 2022

@author: mahathi
"""
from pathlib import Path
import numpy as np
import prob


#Using ReLu activation function with alpha=0 and beta=1

alpha=0 
beta=1


#lip.params

form='neuron' #Other forms: neuron, network, layer, network-rand, network-dec-vars

#weight path of barrier and controller

barr_weight_path = '/home/mahathi/Desktop/Projects/Barrier_NN/FAoC-tool-master/verify/pendulum_lmi/saved_weights/barr_nn.mat'
ctrl_weight_path = '/home/mahathi/Desktop/Projects/Barrier_NN/FAoC-tool-master/verify/pendulum_lmi/saved_weights/ctrl_nn.mat'


#Only required for specific forms 
split=True #split and solve or not?
parallel=False #parallelize the computations? 
verbose=False #prints CVX output if true
split_size=2 #only if split is True
num_neurons=100; #number of neurons to couple for LipSDP-Network-rand formulation
num_workers=0; #number of workers for parallelization of splitting formulations
num_decision_vars=10; #specify number of decision variables to be used for LipSDP

#For the quadratic form representation of sets. Only works for  

domain=np.array(prob.DOMAIN)
init=np.array(prob.INIT)
uns1=np.array(prob.UNSAFE1)
uns2=np.array(prob.UNSAFE2)
uns3=np.array(prob.UNSAFE3)
uns4=np.array(prob.UNSAFE4)

x_min=domain[:,0]
x_max=domain[:,1]

# xi_min=init[:,0]
# xi_max=init[:,1]

# xu_min=uns[:,0]
# xu_max=uns[:,1]

dim=len(domain)