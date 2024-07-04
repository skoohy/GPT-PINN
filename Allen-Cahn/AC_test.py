from AC_precompute import Pt_lPxx_eP
from AC_optimizer import grad_descent
from AC_models import GPT
import numpy as np
import torch
import time 

device = torch.device("cuda")

###############################################################################

def gpt_test(ac_test, out, out_t, out_xx, out_IC, fhat, xt_len, IC_size, BC_size, 
             IC_u, c_initial, epochs_gpt, lr_gpt, U_test, out_BC_ub, out_BC_lb, 
             out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x):
    
    times    = np.zeros(len(ac_test))
    gpt_soln = np.zeros((U_test.shape[0], len(ac_test)))
    
    for i, ac_param in enumerate(ac_test):
        t_start = time.time()
        
        lmbda, eps = ac_param
    
        Pt_lPxx_eP_term =  Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps)
        
        GD = grad_descent(lmbda, eps, xt_len, IC_size, BC_size, IC_u, Pt_lPxx_eP_term, 
                          out, out_IC, out_BC_ub, out_BC_lb, out_BC_diff, 
                          out_BC_ub_x, out_BC_lb_x, out_BC_diff_x, 
                          epochs_gpt, lr_gpt)
        
        c = c_initial
        c_reshaped = c.unsqueeze(1)
        for j in range(1, epochs_gpt+1):            
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
        
        soln = torch.matmul(U_test, c_reshaped)
                
        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
            
        gpt_soln[:,i][:,None] = soln.detach().cpu().numpy()
    return times, gpt_soln

###############################################################################

def gpt_test_loss(ac_test, out, out_t, out_xx, out_IC, fhat, xt_len, IC_size, BC_size, 
                  IC_u, c_initial, epochs_gpt, lr_gpt, U_test, out_BC_ub, out_BC_lb, 
                  out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x):
    
    losses = np.zeros((epochs_gpt+1, len(ac_test)))
    
    for i, ac_param in enumerate(ac_test):                
        lmbda, eps = ac_param
    
        Pt_lPxx_eP_term =  Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps)
        
        GD = grad_descent(lmbda, eps, xt_len, IC_size, BC_size, IC_u, Pt_lPxx_eP_term, 
                          out, out_IC, out_BC_ub, out_BC_lb, out_BC_diff, 
                          out_BC_ub_x, out_BC_lb_x, out_BC_diff_x, 
                          epochs_gpt, lr_gpt)

        GPT_NN = GPT(lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, out_BC_ub_x, 
                     out_BC_lb_x, fhat, Pt_lPxx_eP_term, IC_u)
        
        c = c_initial
        c_reshaped = c.unsqueeze(1)
        losses[0,i] = GPT_NN.loss(c_reshaped)
        for j in range(1, epochs_gpt+1):            
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            losses[j,i] = GPT_NN.loss(c_reshaped)
    return losses