import torch.nn as nn
import torch

device = torch.device("cuda")

def GPT_residual(c, eps, out, Pt_lPxx_eP_term):
    u          = torch.matmul(out, c)
    ut_luxx_eu = torch.matmul(Pt_lPxx_eP_term, c)
    eu3        = torch.mul(eps, torch.pow(u,3))
    return torch.add(ut_luxx_eu, eu3)

class GPT(nn.Module):
    def __init__(self, lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, 
                 out_BC_ub_x, out_BC_lb_x, fhat, Pt_lPxx_eP_term, IC_u):
        super().__init__()        
        self.lmbda = lmbda
        self.eps   = eps
        
        self.loss_function = nn.MSELoss(reduction='mean')

        self.IC_u            = IC_u
        self.fhat            = fhat
        self.Pt_lPxx_eP_term = Pt_lPxx_eP_term
        self.out_IC          = out_IC
        self.out_BC_ub       = out_BC_ub
        self.out_BC_lb       = out_BC_lb
        self.out             = out
        self.out_BC_ub_x     = out_BC_ub_x
        self.out_BC_lb_x     = out_BC_lb_x

    def lossR(self, c):
        u          = torch.matmul(self.out, c)
        ut_luxx_eu = torch.matmul(self.Pt_lPxx_eP_term, c)
        eu3        = torch.mul(self.eps, torch.pow(u,3))
        f          = torch.add(ut_luxx_eu, eu3)
        return self.loss_function(f, self.fhat)
    
    def lossBC(self, c):
        return self.loss_function(torch.matmul(self.out_BC_ub, c), 
                                  torch.matmul(self.out_BC_lb, c))
    
    def lossBC_x(self, c):
        return self.loss_function(torch.matmul(self.out_BC_ub_x, c), 
                                  torch.matmul(self.out_BC_lb_x, c))
    
    def lossIC(self, c):
        return self.loss_function(torch.matmul(self.out_IC, c), self.IC_u)
 
    def loss(self, c):
        loss_R    = self.lossR(c)
        loss_IC   = self.lossIC(c)
        loss_BC   = self.lossBC(c)
        loss_BC_x = self.lossBC_x(c)
        return loss_R + loss_IC + loss_BC + loss_BC_x
    
class P(nn.Module):
    def __init__(self, w, b, layers):
        super().__init__()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])

        for i in range(len(w)-1):
            self.linears[i].weight.data = w[i]        
        self.linears[-1].weight.data = torch.Tensor(w[-1]).view(1,self.layers[-2])
        
        for i in range(len(b)-1):
            self.linears[i].bias.data = b[i]
        self.linears[-1].bias.data = torch.Tensor(b[-1]).view(-1)
        
        self.activation = nn.Tanh()
        
    def forward(self, x):       
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)        
        a = self.linears[-1](a)
        return a