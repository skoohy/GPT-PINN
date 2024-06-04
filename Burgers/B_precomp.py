import torch.autograd as autograd
import torch

device = torch.device("cuda")

def derivatives(xt, out):
    out_xt = autograd.grad(out, xt, torch.ones_like(out).to(device),
                         create_graph=True)[0]
            
    out_xx_tt = autograd.grad(out_xt, xt, 
                              torch.ones_like(out_xt).to(device))[0] 

    out_x = out_xt[:,0].detach().unsqueeze(1)
    out_t = out_xt[:,1].detach().unsqueeze(1)
    out_xx = out_xx_tt[:,0].unsqueeze(1)
    return out_t, out_x, out_xx

def Pt_nu_P_xx(nu, out_t, out_xx):
    t1 = torch.mul(-nu, out_xx)
    return torch.add(t1, out_t)

def inputs(PINN, xt, out, out_t, out_x, out_xx, out_IC, out_BC, IC_xt, BC_xt, 
           i, out_test, xt_test, xt_size, num_mag, idx_list):
    
    end = i+1
    P = PINN(xt)
    out[:,i][:,None] = P.detach()
    P_t, P_x, P_xx = derivatives(xt, P) 
    out_x[:,i][:,None]  = P_x
    out_xx[:,i][:,None] = P_xx
    out_t[:,i][:,None]  = P_t
        
    val, index      = torch.sort(torch.abs(P_xx.view(-1)))
    largest_indices = torch.LongTensor(index[xt_size-num_mag:xt_size].cpu())
    idx_list[i]     = largest_indices
    
    out_t[:,i][:,None] .put_(idx_list[i].to(device), 
                             torch.zeros(num_mag).to(device))
    
    out_x[:,i][:,None] .put_(idx_list[i].to(device), 
                             torch.zeros(num_mag).to(device))
    
    out_xx[:,i][:,None].put_(idx_list[i].to(device), 
                             torch.zeros(num_mag).to(device))
    
    out[:,i][:,None]   .put_(idx_list[i].to(device), 
                             torch.zeros(num_mag).to(device))
    
    out_IC[:,i][:,None] = PINN(IC_xt).detach()
    out_BC[:,i][:,None] = PINN(BC_xt).detach()
    
    out_test[:,i][:,None] = PINN(xt_test).detach()
    
    return out[:,0:end], out_x[:,0:end], out_t[:,0:end], out_xx[:,0:end],\
           out_IC[:,0:end], out_BC[:,0:end]