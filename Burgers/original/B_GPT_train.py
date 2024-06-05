import torch
from B_GPT_optimizer import grad_descent
torch.set_default_dtype(torch.float)

def gpt_train(GPT_PINN, nu, xt_resid, IC_xt, IC_u, BC_xt, BC_u,
              P_resid_values, P_IC_values, P_BC_values, Pt_nu_P_xx_term,
              P_x_term, epochs_gpt, lr_gpt,
              largest_loss=None, largest_case=None, testing=False):
    
    GD = grad_descent(nu, xt_resid, IC_xt, BC_xt, IC_u, P_resid_values, P_IC_values, 
                      P_BC_values, P_x_term, Pt_nu_P_xx_term, lr_gpt)

    if (testing == False): 
        loss_values = GPT_PINN.loss()
        for i in range(1, epochs_gpt+1):
            if (loss_values < largest_loss): 
                break
                
            else:
                c = GPT_PINN.linears[1].weight.data.view(-1)
                GPT_PINN.linears[1].weight.data = GD.update(c)
                
                if (i == epochs_gpt):
                    largest_case = nu
                    largest_loss = GPT_PINN.loss() 
                    
            loss_values = GPT_PINN.loss()
        return largest_loss, largest_case
    
    elif (testing):
        for i in range(1, epochs_gpt+1):
            c = GPT_PINN.linears[1].weight.data.view(-1)
            GPT_PINN.linears[1].weight.data = GD.update(c)