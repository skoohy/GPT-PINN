import torch
from KG_GPT_optimizer import grad_descent
torch.set_default_dtype(torch.float)

def gpt_train(GPT_PINN, alpha, beta, gamma, xt_resid, 
              IC_xt, IC_u1, IC_u2, BC_xt, BC_u, xcos_x2cos2_term,
              Ptt_aPxx_bP_term, alpha_P_xx_term, 
              beta_P_term,      gamm2_P_term,
              P_resid_values,   P_IC_values, P_BC_values, Pi_t_term,
              network_gradients, epochs_gpt, lr_gpt, largest_loss, largest_case):
    
    GD = grad_descent(alpha, beta, gamma, xt_resid, IC_xt, BC_xt, IC_u1, IC_u2, 
                      BC_u, xcos_x2cos2_term, 
                      Ptt_aPxx_bP_term, alpha_P_xx_term, beta_P_term, 
                      gamm2_P_term, P_resid_values, P_IC_values, P_BC_values, 
                      Pi_t_term, lr_gpt, network_gradients)
    
    loss_values = GPT_PINN.loss().item()
    for i in range(1, epochs_gpt+1):
        if (loss_values < largest_loss): 
            break
            
        else:
            c = GPT_PINN.linears[1].weight.data.view(-1)
            GPT_PINN.linears[1].weight.data = GD.update(c)
            
            if (i == epochs_gpt):
                largest_case = (alpha, beta, gamma)
                largest_loss = GPT_PINN.loss().item() 
                
        loss_values = GPT_PINN.loss()
                    
    return largest_loss, largest_case
        
