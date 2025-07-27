#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:11:10 2022

@author: mahathi
"""

# Defining safe and unsafe sets- vector field

import torch
import superp_init as superp
import math


############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[-1 / 15 * math.pi, 1 / 15 * math.pi], \
            [- 1 / 15 * math.pi, 1 / 15 * math.pi]
        ]
INIT_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_INIT = []
SUB_INIT_SHAPE = []


UNSAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

UNSAFE1=[[-1 / 4 * math.pi, -1 / 6 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]]
UNSAFE2=[[1 / 6 * math.pi, 1 / 4 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]]
UNSAFE3=[[-1 / 4 * math.pi, 1 / 4 * math.pi], [-1 / 4 * math.pi, -1 / 6 * math.pi]]
UNSAFE4=[[-1 / 4 * math.pi, 1 / 4 * math.pi], [1 / 6 * math.pi, 1 / 4 * math.pi]]


#SUB_UNSAFE = [[-1 / 4 * math.pi, -1 / 6 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]], \
 #                   [[1 / 6 * math.pi, 1 / 4 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]], \
  #                      [[-1 / 4 * math.pi, 1 / 6 * math.pi], [-1 / 4 * math.pi, -1 / 6 * math.pi]], \
   #                         [[-1 / 4 * math.pi, 1 / 6 * math.pi], [1 / 6 * math.pi, 1 / 4 * math.pi]]

#SUB_UNSAFE_SHAPE = [1, 1, 1, 1]

# the the domain in super-rectangle
DOMAIN = [[-1 / 4 * math.pi, 1 / 4 * math.pi], \
            [-1 / 4 * math.pi, 1 / 4 * math.pi]
        ]
DOMAIN_SHAPE = 1 # 1 for rectangle



############################################
# set the range constraints
############################################
# accept a two-dimensional tensor and return a 
# tensor of bool with the same number of columns
def cons_init(x): 
    ini= (x[:,0] >= INIT[0][0]) & (x[:,0 ] <= INIT[0][1]) & (x[:,1] >= INIT[1][0]) & (x[:,1] <= INIT[1][1])
    return ini

def cons_unsafe1(x):
    uns1=(x[:,0] >= UNSAFE1[0][0]) & (x[:,0 ] <= UNSAFE1[0][1]) & (x[:,1] >= UNSAFE1[1][0]) & (x[:,1] <= UNSAFE1[1][1])
    return uns1

def cons_unsafe2(x):
    uns2= (x[:,0] >= UNSAFE2[0][0]) & (x[:,0 ] <= UNSAFE2[0][1]) & (x[:,1] >= UNSAFE2[1][0]) & (x[:,1] <= UNSAFE2[1][1])
    return  uns2

def cons_unsafe3(x):
    uns3=(x[:,0] >= UNSAFE3[0][0]) & (x[:,0 ] <= UNSAFE3[0][1]) & (x[:,1] >= UNSAFE3[1][0]) & (x[:,1] <= UNSAFE3[1][1])
    return uns3

def cons_unsafe4(x):
    uns4= (x[:,0] >= UNSAFE4[0][0]) & (x[:,0 ] <= UNSAFE4[0][1]) & (x[:,1] >= UNSAFE4[1][0]) & (x[:,1] <= UNSAFE4[1][1])
    return  uns4
 
def cons_domain(x):
    dom1= (x[:,0] >= -math.pi/4) & (x[:,0] <= -math.pi/30) & (x[:,1] >= -math.pi/4) & (x[:,1] <= -math.pi/30)
    dom2= (x[:,0] >= math.pi/30) & (x[:,0] <= math.pi/4) & (x[:,1] >= math.pi/30) & (x[:,1] <= math.pi/4)
    dom3= (x[:,0] >= math.pi/30) & (x[:,0] <= math.pi/4) & (x[:,1] >= -math.pi/4) & (x[:,1] <= -math.pi/30)
    dom4= (x[:,0] >= -math.pi/4) & (x[:,0] <= -math.pi/30) & (x[:,1] >= math.pi/30) & (x[:,1] <= math.pi/4)
    return dom1 | dom2 | dom3 | dom4



   #return x[:, 0] == x[:, 0] # equivalent to True


############################################
# set the vector field
############################################
# this function accepts a tensor input and returns the vector field of the same size
def vector_field(x, ctrl_nn):
    # the vector of functions
    tau=0.01
    def f(i, x):
        if i == 1:
            return x[:,0]+ tau*x[:, 1] # x[:, 1] stands for x2
        elif i == 2:
            return x[:,1]+tau*(9.8 * (x[:, 0] - torch.pow(x[:, 0], 3) / 6.0) + (ctrl_nn(x))[:, 0])
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(superp.DIM_S)], dim=1)
    return vf

L_x=1.5
L_u=0.01
#L_f=L_x+L_u*L_c