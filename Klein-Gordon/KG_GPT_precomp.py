import torch
from torch import cos
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autograd_calculations(xt_resid, P):
    """Compute graidents w.r.t xx and tt for the residual data"""
    xt_resid = xt_resid.to(device).requires_grad_()
    Pi = P(xt_resid).to(device)
    P_xt = autograd.grad(Pi, xt_resid, torch.ones(xt_resid.shape[0], 1).to(device), create_graph=True)[0]
    P_xx_tt = autograd.grad(P_xt, xt_resid, torch.ones(xt_resid.shape).to(device), create_graph=True)[0] 

    P_xx = P_xx_tt[:,[0]]
    P_tt = P_xx_tt[:,[1]]
    return P_xx, P_tt

def Pi_t(IC_xt, P):
    IC_xt = IC_xt.requires_grad_()
    P_IC_values = P(IC_xt).to(device)
    P_t = autograd.grad(P_IC_values, IC_xt, torch.ones(IC_xt.shape[0], 1).to(device), 
                          create_graph=True)[0][:,[1]].detach()
    return P_t

def Ptt_aPxx_bP(alpha, beta, P_tt, P_xx, P_resid_values):
    """Ptt * alpha*Pxx + beta*P"""
    t1 = P_tt
    t2 = torch.mul(alpha,P_xx)
    t3 = torch.mul(beta,P_resid_values)
    return torch.add(torch.add(t1,t2),t3)

def alpha_times_P_xx(alpha, P_xx_term):
    """alpha*Pxx"""    
    a_P_xx = torch.mul(alpha,P_xx_term)
    return a_P_xx

def beta_times_P(beta, P_resid_values):
    """beta*P"""
    b_P = torch.mul(beta,P_resid_values)
    return b_P

def gamma2_P(gamma, P_resid_values):
    """2*gamma*P"""    
    g2_P = torch.mul(2*gamma, P_resid_values)
    return g2_P

def xcos_x2cos2(x_resid, t_resid):
    """x*cos(t) - x^2*cos^2(t)"""
    return x_resid*cos(t_resid) - (x_resid**2)*(cos(t_resid)**2)


