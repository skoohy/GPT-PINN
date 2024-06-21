import torch.autograd as autograd
from torch import cos
import torch

device = torch.device("cuda")

def second_derivatives(xt, out):
    out_xt = autograd.grad(out, xt, torch.ones_like(out).to(device),
                         create_graph=True)[0]
            
    out_xx_tt = autograd.grad(out_xt, xt, 
                              torch.ones_like(out_xt).to(device))[0] 
        
    out_xx = out_xx_tt[:,0].unsqueeze(1)
    out_tt = out_xx_tt[:,1].unsqueeze(1)
    return out_xx, out_tt

def initial_derivative(IC_xt, out_IC):
    out_t = autograd.grad(out_IC, IC_xt, 
                          torch.ones_like(out_IC).to(device))[0]
    return out_t[:,1].unsqueeze(1)

def Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out):
    t1 = torch.mul(alpha, out_xx)
    t2 = torch.mul(beta, out)
    return torch.add(torch.add(out_tt, t1), t2)

def gamma2_P(gamma, out):
    return torch.mul(2*gamma, out)

def xcos_term(x, t):
    return x*cos(t) - (x**2)*(cos(t)**2)

def inputs(PINN, xt, out, out_xx, out_tt, out_IC_t,
           out_IC, out_BC, IC_xt, BC_xt, i, out_test, xt_test):
    
    end = i+1
    P = PINN(xt)
    out[:,i][:,None] = P.detach()
    P_xx, P_tt = second_derivatives(xt, P) 
    out_xx[:,i][:,None] = P_xx
    out_tt[:,i][:,None] = P_tt
    
    P_IC = PINN(IC_xt)
    out_IC[:,i][:,None]   = P_IC.detach()
    out_IC_t[:,i][:,None] = initial_derivative(IC_xt, P_IC)
    out_BC[:,i][:,None]   = PINN(BC_xt).detach()
    
    out_test[:,i][:,None] = PINN(xt_test).detach()
    
    return out[:,0:end], out_xx[:,0:end], out_tt[:,0:end], out_IC_t[:,0:end],\
           out_IC[:,0:end], out_BC[:,0:end]
    
    
    
    
    
    
    