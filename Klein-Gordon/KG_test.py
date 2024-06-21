from KG_precomp import Ptt_aPxx_bP, gamma2_P
from KG_optimizer import grad_descent
from KG_models import GPT, NN
import numpy as np
import torch
import time

device = torch.device("cuda")

def gpt_test(kg_test, out, out_xx, out_tt, out_IC, out_IC_t, out_BC, fhat, 
             xcos, xt_len, IC_size, BC_size, IC_u1, IC_u2, BC_u, c_initial, 
             epochs_gpt, lr_gpt, U_test):
    
    times    = np.zeros(len(kg_test))
    gpt_soln = np.zeros((U_test.shape[0], len(kg_test)))
    
    for i, kg_param in enumerate(kg_test):
        t_start = time.time()
        
        alpha, beta, gamma = kg_param
        
        Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out)
        gamma2_P_term    = gamma2_P(gamma, out)
        
        GD = grad_descent(alpha, beta, gamma, xt_len, IC_size, BC_size, IC_u1, 
                          IC_u2, BC_u, xcos, Ptt_aPxx_bP_term, gamma2_P_term, 
                          out, out_IC, out_IC_t, out_BC, epochs_gpt, lr_gpt)
        
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
###############################################################################

def gpt_test_loss(kg_test, out, out_xx, out_tt, out_IC, out_IC_t, out_BC, fhat, 
                  xcos, xt_len, IC_size, BC_size, IC_u1, IC_u2, BC_u, 
                  c_initial, epochs_gpt, lr_gpt):
    
    losses = np.zeros((11, len(kg_test)))
    
    for i, kg_param in enumerate(kg_test):        
        alpha, beta, gamma = kg_param
        
        Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out)
        gamma2_P_term    = gamma2_P(gamma, out)
        
        GD = grad_descent(alpha, beta, gamma, xt_len, IC_size, BC_size, IC_u1, 
                          IC_u2, BC_u, xcos, Ptt_aPxx_bP_term, gamma2_P_term, 
                          out, out_IC, out_IC_t, out_BC, epochs_gpt, lr_gpt)
        
        GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos, fhat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u)
        
        c = c_initial
        c_reshaped = c.unsqueeze(1)
        losses[0,i] = GPT_NN.loss(c_reshaped)
        for j in range(1, epochs_gpt+1):            
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            if (j % 500 == 0):
                losses[int(j/500),i] = GPT_NN.loss(c_reshaped)
    return losses

###############################################################################
###############################################################################

def pinn_test(kg_test, layers_pinn, xcos_x2cos2, xt_resid, IC_xt, IC_u1, IC_u2, 
              BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, xt_test):

    times     = np.zeros(len(kg_test))
    pinn_soln = np.zeros((xt_test.shape[0], len(kg_test)))

    for i, kg_param in enumerate(kg_test):  
        t_start = time.time()
        
        alpha, beta, gamma = kg_param
        
        PINN = NN(layers_pinn, alpha, beta, gamma, xcos_x2cos2).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        soln = PINN(xt_test)        

        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
            
        pinn_soln[:,i][:,None] = soln.detach().cpu().numpy()
    return times, pinn_soln

###############################################################################
###############################################################################

def pinn_test_loss(kg_test, layers_pinn, xcos_x2cos2, xt_resid, IC_xt, IC_u1, 
                   IC_u2, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn):

    losses = np.zeros((101, len(kg_test)))

    for i, kg_param in enumerate(kg_test):          
        alpha, beta, gamma = kg_param
        
        PINN = NN(layers_pinn, alpha, beta, gamma, xcos_x2cos2).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        losses[0,i] = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1,
                                IC_u2, BC_xt, BC_u).item()
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (j % 1000 == 0) or (j == epochs_pinn):
                losses[int(j/1000),i] = PINN.loss(xt_resid, f_hat, IC_xt, 
                                        IC_u1, IC_u2, BC_xt, BC_u).item()
    return losses