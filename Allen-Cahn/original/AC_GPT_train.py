import torch
from AC_GPT_optimizer import grad_descent
torch.set_default_dtype(torch.float)

def gpt_train(GPT_PINN, lmbda, eps, xt_resid, IC_xt, BC_xt, IC_u, BC_u,
              P_resid_values, P_IC_values, 
              P_BC_values, Pt_lPxx_eP_term,
              lr_gpt, epochs_gpt, largest_loss=None, largest_case=None,
              testing=False):
    
    GD = grad_descent(lmbda, eps, xt_resid, IC_xt, BC_xt, IC_u, BC_u,
                 P_resid_values, P_IC_values, P_BC_values, Pt_lPxx_eP_term, lr_gpt)

    if (testing == False): 
        loss_values = GPT_PINN.loss()
        for i in range(1, epochs_gpt+1):
            if (loss_values < largest_loss): 
                break
                
            else:
                c = GPT_PINN.linears[1].weight.data.view(-1)
                GPT_PINN.linears[1].weight.data = GD.update(c)
                
                if (i == epochs_gpt):
                    largest_case = [lmbda, eps]
                    largest_loss = GPT_PINN.loss() 
                    
            loss_values = GPT_PINN.loss()
        return largest_loss, largest_case
    
    elif (testing):
        for i in range(1, epochs_gpt+1):
            c = GPT_PINN.linears[1].weight.data.view(-1)
            GPT_PINN.linears[1].weight.data = GD.update(c)