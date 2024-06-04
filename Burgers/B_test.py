from B_optimizer import grad_descent
from B_precomp import Pt_nu_P_xx
from B_models import GPT, NN
import numpy as np
import torch
import time

device = torch.device("cuda")

###############################################################################
###############################################################################

def gpt_test(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons, out_test):
    
    times    = np.zeros(len(nu_train))
    gpt_soln = np.zeros((out_test.shape[0], len(nu_train)))
    
    neuron_cnt = out.shape[1]; neuron_params = neurons
    
    dist      = np.zeros(neuron_cnt) 
    c_initial = torch.zeros(neuron_cnt).to(device)
    
    for i, nu in enumerate(nu_train):
        t_start = time.time()
        
        for k, nu_neuron in enumerate(neuron_params): 
            dist[k] = np.abs(nu_neuron - nu)

        d      = np.argsort(dist) 
        first  = d[0] 
        second = d[1] 

        a      = dist[first]
        b      = dist[second] 
        bottom = a+b
        
        c_initial[:]      = 0
        c_initial[first]  = b / bottom 
        c_initial[second] = a / bottom
            
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
    
        GD = grad_descent(nu, xt_size, IC_size, BC_size, IC_u, out, out_IC, 
                          out_BC, out_x, Pt_nu_P_xx_term, lr_gpt)
                
        c = c_initial
        c_reshaped = c.unsqueeze(1)
        for j in range(1, epochs_gpt+1):
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
                                            
        soln = torch.matmul(out_test, c_reshaped)
        
        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
        
        gpt_soln[:,i][:,None] = soln.cpu().numpy()
    return times, gpt_soln

###############################################################################
###############################################################################

def gpt_test_loss(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons):
    
    losses = np.zeros((11, len(nu_train)))
    
    neuron_cnt = out.shape[1]; neuron_params = neurons
    
    dist      = np.zeros(neuron_cnt) 
    c_initial = torch.zeros(neuron_cnt).to(device)
    
    for i, nu in enumerate(nu_train): 
        for k, nu_neuron in enumerate(neuron_params): 
            dist[k] = np.abs(nu_neuron - nu)

        d      = np.argsort(dist) 
        first  = d[0] 
        second = d[1] 

        a      = dist[first]
        b      = dist[second] 
        bottom = a+b
        
        c_initial[:]      = 0
        c_initial[first]  = b / bottom 
        c_initial[second] = a / bottom
            
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
    
        GD = grad_descent(nu, xt_size, IC_size, BC_size, IC_u, out, out_IC, 
                          out_BC, out_x, Pt_nu_P_xx_term, lr_gpt)
        
        GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat).to(device)
        
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

def pinn_test(b_test, layers_pinn, xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat, 
              epochs_pinn, lr_pinn, xt_test, tol):

    times     = np.zeros(len(b_test))
    pinn_soln = np.zeros((xt_test.shape[0], len(b_test)))

    for i, b_param in enumerate(b_test):  
        t_start = time.time()
        
        nu = b_param
        
        PINN = NN(layers_pinn, nu).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat) 
            
            if (loss < tol):
                break
            
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

def pinn_test_loss(b_test, layers_pinn, xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                   f_hat, epochs_pinn, lr_pinn, tol):

    losses = np.zeros((61, len(b_test)))
    
    for i, b_param in enumerate(b_test):          
        nu = b_param
        
        PINN = NN(layers_pinn, nu).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        losses[0,i] = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                                f_hat).item()
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat) 
            
            if (j % 1000 == 0) or (j == epochs_pinn):
                losses[int(j/1000),i] = loss.item()
            
            if (loss < tol):
                losses[np.where(losses[:,i] == 0)[0][0],i] = loss.item()
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses