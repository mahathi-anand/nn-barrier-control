#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:15:42 2022

@author: mahathi
"""

''' This code is for checking the satisfaction of barrier conditions by the neural network '''

"""
 Created on Tue May 24 17:01:18 2022

 @author: mahathi
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch 
import args
from matplotlib import cm
import prob
import math
from mpl_toolkits.mplot3d import Axes3D


def sysdyn(x,ctrl_nn):
    tau=0.01;
    f=np.empty([1,2])
    xt=torch.tensor(x)
    f[0,0]=xt[0]+ tau*xt[1] 
    f[0,1]=xt[1]+tau*(9.8 * (x[0] - torch.pow(xt[0], 3) / 6.0) + ctrl_nn(xt).detach().numpy())
    return f

barr_nn=torch.load('saved_weights/barr_nn')
ctrl_nn=torch.load('saved_weights/ctrl_nn')

s=20;
t=8000;


x=np.empty([t,2])

#plot the system 
A=[];
for j in range(s):
    x[0,0]=args.init[0][0]+(args.init[0][1]-args.init[0][0])*np.random.rand(1,1)
    x[0,1]=args.init[1][0]+(args.init[1][1]-args.init[1][0])*np.random.rand(1,1)
    for i in range(t-1):
        x[i+1,:]=sysdyn(x[i,:],ctrl_nn)
        if x[i+1,0] >= args.x_max[0]:
            x[i+1,0] = args.x_max[0]
        elif x[i+1,0] <= args.x_min[0]:
            x[i+1,0] = args.x_min[0]
        if x[i+1,1] >= args.x_max[1]:
            x[i+1,1] = args.x_max[1]
        elif x[i+1,1] <= args.x_min[1]:
            x[i+1,1] = args.x_min[1]    
        A.append(barr_nn(torch.tensor(x[i,:])).detach().numpy())    
    
    #axs.stairs(x[:,0],np.array(range(0,t+1)))
    #axs.stairs(x[:,1],np.array(range(0,t+1)))
    #plt.show()
        
plt.plot(x[:,0],x[:,1])   
plt.xlabel("x1")
plt.ylabel("x2")
    
    
 #plot the barrier conditions 
 
#last condition
 
dx1=0.01
dx2=0.01
 
x1, x2 = np.mgrid[slice(args.domain[0][0], args.domain[0][1]+dx1, dx1), slice(args.domain[1][0], args.domain[1][1]+dx2, dx2)]

# x1.shape=x1.shape[0]*x1.shape[1]
# x2.shape=x2.shape[0]*x2.shape[1]
x1d = np.reshape(x1, (1, x1.size))
x2d = np.reshape(x2, (1, x2.size))
xd=np.concatenate((x1d.T,x2d.T),1)

B=barr_nn(torch.tensor(xd))
Bf=barr_nn(torch.tensor(prob.vector_field(torch.tensor(xd),ctrl_nn)))
cond3=np.transpose((Bf-B).detach().numpy())
#cond3=np.transpose(ctrl_nn(torch.tensor(xd)).detach().numpy())
cond3=np.reshape(cond3,x1.shape)

fig,ax=plt.subplots()
im=ax.pcolormesh(x1,x2,cond3, shading='gouraud')
ax.set_xlim(args.domain[0][0],args.domain[0][1]);
ax.set_ylim(args.domain[1][0],args.domain[1][1]);
ax.set_xlabel("x1")
ax.set_ylabel("x2")
fig.suptitle("Last Condition")
fig.colorbar(im, shrink=0.5, aspect=5)

plt.show()

#first condition
dx1=0.01
dx2=0.01
x1, x2 = np.mgrid[slice(args.init[0][0], args.init[0][1]+dx1, dx1), slice(args.init[1][0], args.init[1][1]+dx2, dx2)]
x1d = np.reshape(x1, (1, x1.size))
x2d = np.reshape(x2, (1, x2.size))
xd=np.concatenate((x1d.T,x2d.T),1)

B=barr_nn(torch.tensor(xd)).detach().numpy()
B=np.reshape(B,x1.shape)

fig,ax1=plt.subplots()
im=ax1.pcolormesh(x1,x2,B, shading='gouraud')
ax1.set_xlim(args.init[0][0],args.init[0][1]);
ax1.set_ylim(args.init[1][0],args.init[1][1]);
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
fig.suptitle("Initial Condition")
fig.colorbar(im, shrink=0.5, aspect=5)

plt.show()

#second condition unsafe1

dx1=0.01
dx2=0.01
x1, x2 = np.mgrid[slice(args.uns1[0][0], args.uns1[0][1]+dx1, dx1), slice(args.uns1[1][0], args.uns1[1][1]+dx2, dx2)]
x1d = np.reshape(x1, (1, x1.size))
x2d = np.reshape(x2, (1, x2.size))
xd=np.concatenate((x1d.T,x2d.T),1)

B=barr_nn(torch.tensor(xd)).detach().numpy()
B=np.reshape(B,x1.shape)

fig,ax2=plt.subplots()
im=ax2.pcolormesh(x1,x2, B, shading='gouraud',cmap='inferno')
ax2.set_xlim(args.uns1[0][0],args.uns1[0][1]);
ax2.set_ylim(args.uns1[1][0],args.uns1[1][1]);
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
fig.suptitle("Unsafe Condition 1")
fig.colorbar(im, shrink=0.5, aspect=5)

plt.show()

#second condition unsafe2 

dx1=0.01
dx2=0.01
x1, x2 = np.mgrid[slice(args.uns2[0][0], args.uns2[0][1]+dx1, dx1), slice(args.uns2[1][0], args.uns2[1][1]+dx2, dx2)]
x1d = np.reshape(x1, (1, x1.size))
x2d = np.reshape(x2, (1, x2.size))
xd=np.concatenate((x1d.T,x2d.T),1)

B=barr_nn(torch.tensor(xd)).detach().numpy()
B=np.reshape(B,x1.shape)

fig,ax3=plt.subplots()
im=ax3.pcolormesh(x1,x2, B, shading='gouraud',cmap='inferno')
ax3.set_xlim(args.uns2[0][0],args.uns2[0][1]);
ax3.set_ylim(args.uns2[1][0],args.uns2[1][1]);
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
fig.suptitle("Unsafe Condition 2")
fig.colorbar(im, shrink=0.5, aspect=5)

plt.show()

#second condition unsafe3

dx1=0.01
dx2=0.01
x1, x2 = np.mgrid[slice(args.uns3[0][0], args.uns3[0][1]+dx1, dx1), slice(args.uns3[1][0], args.uns3[1][1]+dx2, dx2)]
x1d = np.reshape(x1, (1, x1.size))
x2d = np.reshape(x2, (1, x2.size))
xd=np.concatenate((x1d.T,x2d.T),1)

B=barr_nn(torch.tensor(xd)).detach().numpy()
B=np.reshape(B,x1.shape)

fig,ax4=plt.subplots()
im=ax4.pcolormesh(x1,x2, B, shading='gouraud',cmap='inferno')
ax4.set_xlim(args.uns3[0][0],args.uns3[0][1]);
ax4.set_ylim(args.uns3[1][0],args.uns3[1][1]);
ax4.set_xlabel("x1")
ax4.set_ylabel("x2")
fig.suptitle("Unsafe Condition 3")
fig.colorbar(im, shrink=0.5, aspect=5)

plt.show()

#second condition unsafe2 

dx1=0.01
dx2=0.01
x1, x2 = np.mgrid[slice(args.uns4[0][0], args.uns4[0][1]+dx1, dx1), slice(args.uns4[1][0], args.uns4[1][1]+dx2, dx2)]
x1d = np.reshape(x1, (1, x1.size))
x2d = np.reshape(x2, (1, x2.size))
xd=np.concatenate((x1d.T,x2d.T),1)

B=barr_nn(torch.tensor(xd)).detach().numpy()
B=np.reshape(B,x1.shape)

fig,ax5=plt.subplots()
im=ax5.pcolormesh(x1,x2, B, shading='gouraud',cmap='inferno')
ax5.set_xlim(args.uns4[0][0],args.uns4[0][1]);
ax5.set_ylim(args.uns4[1][0],args.uns4[1][1]);
ax5.set_xlabel("x1")
ax5.set_ylabel("x2")
fig.suptitle("Unsafe Condition 4")
fig.colorbar(im, shrink=0.5, aspect=5)

plt.show()

#3d plot of barrier function
mpl.rcParams['legend.fontsize'] = 8

fig=plt.figure()
ax3=fig.gca(projection='3d')
x1=np.linspace(args.domain[0][0], args.domain[0][1], 100)
x2=np.linspace(args.domain[1][0], args.domain[1][1], 100)

B=np.empty(100)
for i in range(len(B)):
    B[i]=barr_nn(torch.tensor(np.array([x1[i],x2[i]])))

ax3.plot(x1,x2,B,label='barrier certificate')
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_zlabel("B(x)")

plt.show()                 
