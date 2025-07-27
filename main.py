import torch
import torch.nn as nn
import superp_init as superp
import ann
import data
import train
import time
#import plot # comment this line if matplotlib, mayavi, or PyQt5 was not successfully installed
#import plot3d

def barr_ctrl_nn():
# generating training model
    barr_nn = ann.gen_nn(superp.N_H_B, superp.D_H_B, superp.DIM_S, 1, superp.BARR_ACT, superp.BARR_IN_BOUND, superp.BARR_OUT_BOUND) # generate the nn model for the barrier
    #ctrl_nn = torch.load('fixed_controller/ctrl_nn') #load pretrained controller
    ctrl_nn = ann.gen_nn(superp.N_H_C, superp.D_H_C, superp.DIM_C, 1,superp.CTRL_ACT, superp.CTRL_IN_BOUND, superp.CTRL_OUT_BOUND) # generate the nn model for the controller

    # loading pre-trained model
    if superp.FINE_TUNE == 1:
        barr_nn=torch.load('barr_nn')
        ctrl_nn=torch.load('ctrl_nn')
    
    
    ###########################################
    # # check counter example
    ###########################################
    # counter_ex = [0.313456207410, -0.35799516543633, -0.059641790995704782]
    # train.test_lie(barr_nn, ctrl_nn, counter_ex)
    # plot3d.plot_sys_3d(barr_nn, ctrl_nn, counter_ex) 
    
    
    # generate training data
    time_start_data = time.time()
    batches_init, batches_unsafe, batches_domain = data.gen_batch_data()
    time_end_data = time.time()
    
    ############################################
    # number of mini_batches
    ############################################
    BATCHES_I = len(batches_init)
    BATCHES_U = len(batches_unsafe)
    BATCHES_D = len(batches_domain)
    BATCHES = max(BATCHES_I, BATCHES_U, BATCHES_D)
    NUM_BATCHES = [BATCHES_I, BATCHES_U, BATCHES_D, BATCHES]
    
    # train and return the learned model
    time_start_train = time.time()
    res = train.itr_train(barr_nn, ctrl_nn, batches_init, batches_unsafe, batches_domain, NUM_BATCHES) 
    time_end_train = time.time()
    
    print("\nData generation totally costs:", time_end_data - time_start_data)
    print("Training totally costs:", time_end_train - time_start_train)
    print("-------------------------------------------------------------------------")
    
    # if res == True:
    #     # save model for fine tuning
    #     torch.save(barr_nn.state_dict(), 'pre_trained_barr.pt')
    #     torch.save(ctrl_nn.state_dict(), 'pre_trained_ctrl.pt')
    #     # generate script for verification in redlog
    #     #redlog.script_gen(barr_nn, ctrl_nn)
    #     # comment this line if matplotlib, mayavi, or PyQt5 was not successfully installed
    #     if superp.VISUAL == 1:
    #         plot.plot_sys(barr_nn, ctrl_nn) 
    # else:
    #     print("Synthesis failed!")
        
    return barr_nn, ctrl_nn


if __name__ =="__main__":
     [barr_nn,ctrl_nn]=barr_ctrl_nn()
     torch.save(barr_nn,r'saved_weights/barr_nn')
     torch.save(ctrl_nn,r'saved_weights/ctrl_nn')
