import torch
from torch import cos
import torch.autograd as autograd
import torch.nn as nn

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class custom_cos(nn.Module):
    """Full PINN Activation Function"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return cos(x)
   
class NN(nn.Module):    
    def __init__(self, layers, alpha, beta, gamma, xcos_x2cos2_term):
        super().__init__()
        
        self.layers = layers
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.xcos_x2cos2_term = xcos_x2cos2_term
        
        for i in range(len(layers)-1):
            nn.init.xavier_uniform_(self.linears[i].weight.data, gain=1)
            #nn.init.xavier_normal_(self.linears[i].weight.data, gain=1)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = custom_cos()
    
    def forward(self, x):       
        a = x.float()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossR(self, xt_residual, f_hat):
        """Residual loss function"""
        g = xt_residual.clone().requires_grad_()

        u = self.forward(g)
        u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        u_xx_tt = autograd.grad(u_xt, g, torch.ones(g.shape).to(device), create_graph=True)[0]
    
        u_xx = u_xx_tt[:,[0]] 
        u_tt = u_xx_tt[:,[1]] 
                        
        f = u_tt + (self.alpha)*u_xx + (self.beta)*(u) + (self.gamma)*(u**2) + self.xcos_x2cos2_term
        return self.loss_function(f, f_hat)
    
    def lossIC1BC(self, ICBC_xt, ICBC_u):
        """First initial and both boundary condition loss function"""
        loss_ICBC = self.loss_function(self.forward(ICBC_xt), ICBC_u)
        return loss_ICBC
    
    def lossIC2(self, IC_xt, IC_u2):
        """Second initial condition loss function"""
        g = IC_xt.clone().requires_grad_()
        
        u = self.forward(g)
        u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        u_t = u_xt[:,[1]]
        
        return self.loss_function(u_t, IC_u2)

    def loss(self, xt_residual, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u):
        """Total loss function"""
        loss_R   = self.lossR(xt_residual, f_hat)
        loss_IC1 = self.lossIC1BC(IC_xt, IC_u1)
        loss_IC2 = self.lossIC2(IC_xt, IC_u2)
        loss_BC  = self.lossIC1BC(BC_xt, BC_u)
                
        return loss_R + loss_IC1 + loss_IC2 + loss_BC 
    
    