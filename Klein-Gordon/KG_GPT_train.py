import torch
from KG_GPT_optimizer import grad_descent
torch.set_default_dtype(torch.float)

def gpt_train(GPT_PINN, alpha, beta, gamma, xt_resid, 
              IC_xt, IC_u1, IC_u2, BC_xt, BC_u, xcos_x2cos2_term,
              Ptt_aPxx_bP_term, alpha_P_xx_term, 
              beta_P_term,      gamm2_P_term,
              P_resid_values,   P_IC_values, P_BC_values, Pi_t_term,
              P_xx_term, P_tt_term, epochs_gpt, lr_gpt, largest_loss, largest_case,
              testing=False):
    
    GD = grad_descent(alpha, beta, gamma, xt_resid, IC_xt, BC_xt, IC_u1, IC_u2, 
                      BC_u, xcos_x2cos2_term, 
                      Ptt_aPxx_bP_term, alpha_P_xx_term, beta_P_term, 
                      gamm2_P_term, P_resid_values, P_IC_values, P_BC_values, 
                      Pi_t_term, P_xx_term, P_tt_term, epochs_gpt, lr_gpt)
    
    
    if testing == False: # Need to comp. loss for training
        loss_values = GPT_PINN.loss()
        for i in range(1, epochs_gpt+1):
            if (loss_values < largest_loss): 
                break
                
            else:
                c = GPT_PINN.linears[1].weight.data.view(-1)
                GPT_PINN.linears[1].weight.data = GD.update(c)
                
                if (i == epochs_gpt):
                    largest_case = [alpha, beta, gamma]
                    largest_loss = GPT_PINN.loss() 
                    
            loss_values = GPT_PINN.loss()
                        
        return largest_loss, largest_case
    
    elif testing == True:
        for i in range(1, epochs_gpt+1): # Don't need to comp. loss for testing
            c = GPT_PINN.linears[1].weight.data.view(-1)
            GPT_PINN.linears[1].weight.data = GD.update(c)
                
            
        
        
