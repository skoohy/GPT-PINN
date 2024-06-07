from B_optimizer import grad_descent
from B_precomp import Pt_nu_P_xx
from B_models import GPT
import numpy as np
import torch

device = torch.device("cuda")

###############################################################################
###############################################################################

def offline_generation(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons, i):
    
    neuron_params = neurons[:i+1]; neuron_cnt = out.shape[1]
    largest_case = 0; largest_loss = 0
    
    if (neuron_cnt == 1):
        c_initial = torch.ones(1).to(device)
        
    if (neuron_cnt != 1):
        dist      = np.zeros(neuron_cnt) 
        c_initial = torch.zeros(neuron_cnt).to(device)

    for nu in nu_train:   
        if (neuron_cnt != 1):
            if nu in neuron_params: 
                index            = np.where(nu == neuron_params) 
                c_initial[:]     = 0
                c_initial[index] = 1
            
            else:
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
        
        brk = False
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
        
        GD = grad_descent(nu, xt_size, IC_size, BC_size, IC_u, out, out_IC, 
                          out_BC, out_x, Pt_nu_P_xx_term, lr_gpt)
                
        GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat).to(device)
                
        c = c_initial
        c_reshaped = c.unsqueeze(1)
        loss = GPT_NN.loss(c_reshaped)
        for j in range(1, epochs_gpt+1):
            if (loss < largest_loss):
                brk = True
                break
                
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            loss = GPT_NN.loss(c_reshaped)
            
        #if (brk == False):
        if (loss > largest_loss):
            largest_case = nu
            largest_loss = loss
    return largest_loss, largest_case

###############################################################################
###############################################################################

def pinn_train(PINN, nu, xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat, 
               epochs_pinn, lr_pinn, tol):
    
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)

    for i in range(1, epochs_pinn+1):
        loss = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat) 
        
        if (loss < tol):
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"PINN Final Loss: {loss.item()}")