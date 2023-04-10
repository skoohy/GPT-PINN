import torch
import torch.autograd as autograd

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autograd_calculations(xt_resid, P):
    """Compute graidents w.r.t t and xx for the residual data"""
    xt_resid = xt_resid.to(device).requires_grad_()
    Pi = P(xt_resid).to(device)
    P_xt = autograd.grad(Pi, xt_resid, torch.ones(xt_resid.shape[0], 1).to(device), create_graph=True)[0]
    P_xx_tt = autograd.grad(P_xt, xt_resid, torch.ones(xt_resid.shape).to(device), create_graph=True)[0] 

    P_t  = P_xt[:,[1]]
    P_xx = P_xx_tt[:,[0]]
    
    return P_t, P_xx

def Pt_lPxx_eP(P_t, P_xx, P, lmbda, eps):
    """Pt - lambda*Pxx - epsilon*P"""
    eP = torch.mul(-eps, P)
    lPxx = torch.mul(-lmbda, P_xx)
    pt_lPxx = torch.add(P_t, lPxx)
    return torch.add(pt_lPxx, eP)