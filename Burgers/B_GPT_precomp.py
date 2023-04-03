import torch
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autograd_calculations(xt_resid, P):
    xt_resid = xt_resid.to(device).requires_grad_()
    Pi = P(xt_resid).to(device)
    P_xt = autograd.grad(Pi, xt_resid, torch.ones(xt_resid.shape[0], 1).to(device), create_graph=True)[0]
    P_xx_tt = autograd.grad(P_xt, xt_resid, torch.ones(xt_resid.shape).to(device), create_graph=True)[0] 

    P_x  = P_xt[:,[0]]
    P_xx = P_xx_tt[:,[0]]
    
    P_t  = P_xt[:,[1]]
    return P_t, P_x, P_xx
    
def Pt_nu_P_xx(nu, P_t, P_xx):
    nu_P_xx = torch.mul(-nu, P_xx)
    return torch.add(P_t, nu_P_xx)