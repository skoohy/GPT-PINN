from AC_models import GPT_residual
import torch.autograd as autograd
import torch

device = torch.device("cuda")

def derivatives(xt, out):
    out_xt = autograd.grad(out, xt, torch.ones_like(out).to(device),
                         create_graph=True)[0]
    out_xx_tt = autograd.grad(out_xt, xt, torch.ones_like(out_xt).to(device))[0] 

    out_t  = out_xt[:,1].detach().unsqueeze(1)
    out_xx = out_xx_tt[:,0].unsqueeze(1)
    return out_t, out_xx 

def boundary_derivative(BC_xt, out_BC):
    out_x = autograd.grad(out_BC, BC_xt, torch.ones_like(out_BC).to(device))[0]
    return out_x[:,0].unsqueeze(1)
    
def Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps):
    eP = torch.mul(-eps, out)
    lPxx = torch.mul(-lmbda, out_xx)
    Pt_lPxx = torch.add(out_t, lPxx)
    return torch.add(Pt_lPxx, eP)
           
def inputs(PINN, xt, out, out_xx, out_t, out_IC, out_BC_ub, out_BC_lb, IC_xt, 
           BC_xt_ub, BC_xt_lb, i, out_test, xt_test, f_hat, xt_size, out_BC_diff,
           out_BC_ub_x, out_BC_lb_x, out_BC_diff_x):
    
    end = i+1
    P = PINN(xt)
    out[:,i][:,None] = P.detach()
    P_t, P_xx = derivatives(xt, P)
    out_t[:,i][:,None]  = P_t
    out_xx[:,i][:,None] = P_xx
    
    out_IC[:,i][:,None] = PINN(IC_xt).detach()
    
    out_ub = PINN(BC_xt_ub)
    out_lb = PINN(BC_xt_lb)
    
    out_BC_ub[:,i][:,None]   = out_ub.detach()
    out_BC_lb[:,i][:,None]   = out_lb.detach()
    out_BC_diff[:,i][:,None] = torch.sub(out_ub.detach(), out_lb.detach())
    
    out_ub_x = boundary_derivative(BC_xt_ub, out_ub)
    out_lb_x = boundary_derivative(BC_xt_lb, out_lb)
    
    out_BC_ub_x[:,i][:,None]   = out_ub_x
    out_BC_lb_x[:,i][:,None]   = out_lb_x
    out_BC_diff_x[:,i][:,None] = torch.sub(out_ub_x, out_lb_x)

    out_test[:,i][:,None] = PINN(xt_test).detach()
    
    return out[:,0:end], out_xx[:,0:end], out_t[:,0:end], out_IC[:,0:end],\
           out_BC_ub[:,0:end], out_BC_lb[:,0:end], out_BC_diff[:,0:end],\
           f_hat, xt_size, out_BC_ub_x[:,0:end], out_BC_lb_x[:,0:end], \
           out_BC_diff_x[:,0:end]