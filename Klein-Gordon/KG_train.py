from KG_precomp import Ptt_aPxx_bP, gamma2_P
from KG_optimizer import grad_descent
from KG_models import GPT
import torch

device = torch.device("cuda")

def offline_generation(kg_train, c_initial, xt_size, IC_size, BC_size, IC_u1, 
                       IC_u2, BC_u, xcos_x2cos2, out, out_xx, out_tt, out_IC, 
                       out_IC_t, out_BC, f_hat, epochs_gpt, lr_gpt):
    
    largest_case = 0; largest_loss = 0
    
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out)
        gamma2_P_term    = gamma2_P(gamma, out)
   
        GD = grad_descent(alpha, beta, gamma, xt_size, IC_size, BC_size, IC_u1, 
                          IC_u2, BC_u, xcos_x2cos2, Ptt_aPxx_bP_term, 
                          gamma2_P_term, out, out_IC, out_IC_t, out_BC, 
                          epochs_gpt, lr_gpt)

        GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2, f_hat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u)
                
        c = c_initial
        c_reshaped = c.unsqueeze(1)
        loss = GPT_NN.loss(c_reshaped)
        for i in range(1, epochs_gpt+1):
            if (loss < largest_loss): 
                break
            
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            loss = GPT_NN.loss(c_reshaped)
            
        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
    
    return largest_loss, largest_case
    
###############################################################################
###############################################################################

def pinn_train(PINN, alpha, beta, gamma, xt_resid, IC_xt, IC_u1, IC_u2, BC_xt, 
               BC_u, f_hat, epochs_pinn, lr_pinn):
        
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
    
    for i in range(1, epochs_pinn+1):
        loss = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"PINN Final Loss: {loss.item()}")