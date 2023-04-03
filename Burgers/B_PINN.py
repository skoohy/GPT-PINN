import torch
import torch.autograd as autograd
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
class NN(nn.Module):    
    def __init__(self, layers, nu):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = nn.Tanh()
    
    def forward(self, x):       
        a = x.float()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossR(self, xt_residual, f_hat):
        g = xt_residual.requires_grad_()

        u = self.forward(g)
        u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        u_xx_tt = autograd.grad(u_xt, g, torch.ones(g.shape).to(device), create_graph=True)[0]
    
        u_xx = u_xx_tt[:,[0]] 
        u_x = u_xt[:,[0]]
        u_t = u_xt[:,[1]] 
                
        f1 = torch.mul(u, u_x)
        f2 = torch.mul(-self.nu, u_xx)
        f3 = torch.add(f1, f2)
        f = torch.add(u_t, f3)
        
        return self.loss_function(f, f_hat)
    
    def lossICBC(self, ICBC_xt, ICBC_u):
        loss_ICBC = self.loss_function(self.forward(ICBC_xt), ICBC_u)
        return loss_ICBC

    def loss(self, xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat):
        loss_R   = self.lossR(xt_resid, f_hat)
        loss_IC = self.lossICBC(IC_xt, IC_u)
        loss_BC  = self.lossICBC(BC_xt, BC_u)
        return loss_R + loss_IC + loss_BC 