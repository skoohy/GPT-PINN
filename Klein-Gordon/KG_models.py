import torch.autograd as autograd
import torch.nn as nn
import torch

device = torch.device("cuda")

class GPT(nn.Module):
    def __init__(self, alpha, beta, gamma, out, out_IC, out_IC_t, 
                 out_BC, xcos, fhat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
    
        self.loss_function = nn.MSELoss(reduction='mean')
        
        self.IC_u1            = IC_u1
        self.IC_u2            = IC_u2
        self.BC_u             = BC_u
        self.fhat             = fhat
        self.out_IC_t         = out_IC_t
        self.xcos             = xcos
        self.Ptt_aPxx_bP_term = Ptt_aPxx_bP_term
        self.out_IC           = out_IC
        self.out_BC           = out_BC
        self.out              = out
               
    def lossR(self, c):
        u  = torch.matmul(self.out, c)
        f1 = torch.matmul(self.Ptt_aPxx_bP_term, c)
        f2 = torch.mul(self.gamma, torch.square(u))
        f3 = torch.add(f1, f2)
        f  = torch.add(f3, self.xcos)
        return self.loss_function(f, self.fhat)
    
    def lossBC(self, c):
        return self.loss_function(torch.matmul(self.out_BC, c), self.BC_u)
    
    def lossIC1(self, c):
        return self.loss_function(torch.matmul(self.out_IC, c), self.IC_u1)
            
    def lossIC2(self, c):
        return self.loss_function(torch.matmul(self.out_IC_t, c), self.IC_u2)
    
    def loss(self, c):
        loss_R   = self.lossR(c)
        loss_IC1 = self.lossIC1(c)
        loss_IC2 = self.lossIC2(c)
        loss_BC  = self.lossBC(c)
        return loss_R + loss_IC1 + loss_IC2 + loss_BC     

###############################################################################
###############################################################################
# PINN

class cos(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cos(x)
   
class NN(nn.Module):    
    def __init__(self, layers, alpha, beta, gamma, xcos_x2cos2):
        super().__init__()
        torch.manual_seed(1234)
        
        self.layers = layers
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        
        self.loss_function = nn.MSELoss(reduction="mean")
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) 
                                      for i in range(len(layers)-1)])
        
        self.xcos_x2cos2 = xcos_x2cos2
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = cos()
    
    def forward(self, x):       
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossR(self, xt, f_hat):
        u = self.forward(xt)
 
        u_xt = autograd.grad(u, xt, torch.ones_like(u).to(device), 
                             create_graph=True)[0]
        
        u_xx_tt = autograd.grad(u_xt, xt, torch.ones_like(u_xt).to(device), 
                                create_graph=True)[0]
        
        u_xx = u_xx_tt[:,0].unsqueeze(1)
        u_tt = u_xx_tt[:,1].unsqueeze(1)
        
        f1 = torch.add(u_tt, torch.mul(self.alpha,u_xx))
        f2 = torch.add(torch.mul(self.beta,u), 
                       torch.mul(self.gamma,torch.square(u)))
        f  = torch.add(torch.add(f1, f2), self.xcos_x2cos2)
        
        return self.loss_function(f, f_hat)
    
    def lossIC1BC(self, ICBC_xt, ICBC_u):
        return self.loss_function(self.forward(ICBC_xt), ICBC_u)
    
    def lossIC2(self, IC_xt, IC_u2):        
        u = self.forward(IC_xt)
        
        u_xt = autograd.grad(u, IC_xt, torch.ones_like(u).to(device), 
                             create_graph=True)[0]
        
        u_t = u_xt[:,1].unsqueeze(1)

        return self.loss_function(u_t, IC_u2)

    def loss(self, xt_residual, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u):
        loss_R   = self.lossR(xt_residual, f_hat)
        loss_IC1 = self.lossIC1BC(IC_xt, IC_u1)
        loss_IC2 = self.lossIC2(IC_xt, IC_u2)
        loss_BC  = self.lossIC1BC(BC_xt, BC_u)
        return loss_R + loss_IC1 + loss_IC2 + loss_BC