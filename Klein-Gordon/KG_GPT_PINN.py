import torch
import torch.nn as nn
import numpy as np
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPT(nn.Module):
    def __init__(self, layers, alpha, beta, gamma, P, initial_c,
                 IC_u1, IC_u2, BC_u, f_hat, xcos_x2cos2_term,
                 activation_resid, activation_IC, activation_BC, 
                 Pi_t_term, P_xx_term, P_tt_term):
        super().__init__()
        self.layers     = layers
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.activation = P
        
        self.loss_function = nn.MSELoss(reduction='mean').to(device)
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1], bias=False) for i in range(len(layers)-1)])
        
        self.IC_u1     = IC_u1
        self.IC_u2     = IC_u2
        self.BC_u      = BC_u
        self.f_hat     = f_hat
        self.Pi_t_term = Pi_t_term
        self.P_xx_term = P_xx_term
        self.P_tt_term = P_tt_term
        
        self.activation_IC     = activation_IC
        self.activation_BC     = activation_BC
        self.activation_resid  = activation_resid
        self.xcos_x2cos2_term  = xcos_x2cos2_term
        
        self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[1].weight.data = torch.Tensor(np.array([initial_c]))
        
    def forward(self, datatype=None, test_data=None):
        if test_data is not None: # Test Data Forward Pass (if test data != resid data)
            a = torch.Tensor().to(device)
            for i in range(0, self.layers[1]):
                a = torch.cat((a, self.activation[i](test_data)), 1)
            final_output = self.linears[-1](a)
            return final_output
        
        if datatype == 'residual': # Residual Data Output
            final_output = self.linears[-1](self.activation_resid).to(device)
            return final_output
        
        if datatype == 'initial': # Initial Data Output
            final_output = self.linears[-1](self.activation_IC).to(device)
            return final_output
        
        if datatype == 'boundary': # Boundary Data Output
            final_output = self.linears[-1](self.activation_BC).to(device)
            return final_output

    def lossR(self):
        """Residual loss function"""
        u = self.forward(datatype='residual')

        cP_tt = torch.matmul(self.P_tt_term, self.linears[1].weight.data[0][:,None])
        cP_xx = torch.matmul(self.P_xx_term, self.linears[1].weight.data[0][:,None])

        #f = cP_tt + (self.alpha)*cP_xx + (self.beta)*(u) + (self.gamma)*(u**2) + self.xcos_x2cos2_term
        f1 = torch.add(cP_tt, self.alpha*cP_xx)
        f2 = torch.add(self.beta*u, self.gamma*torch.square(u))
        f = torch.add(torch.add(f1, f2), self.xcos_x2cos2_term)
        
        return self.loss_function(f, self.f_hat)
    
    def lossIC1BC(self, datatype):
        """First initial and both boundary condition loss function"""
        if datatype=='initial':
            return self.loss_function(self.forward(datatype=datatype), self.IC_u1)
            
        elif datatype=='boundary':
            return self.loss_function(self.forward(datatype=datatype), self.BC_u)
            
    def lossIC2(self):
        """Second initial condition loss function"""
        cP_t = torch.matmul(self.Pi_t_term, self.linears[1].weight.data[0][:,None])

        return self.loss_function(cP_t, self.IC_u2)
    
    def loss(self):
        """Total loss function"""
        loss_R   = self.lossR()
        loss_IC1 = self.lossIC1BC(datatype='initial')
        loss_IC2 = self.lossIC2()
        loss_BC  = self.lossIC1BC(datatype='boundary')
        return loss_R + loss_IC1 + loss_IC2 + loss_BC 
    
    
    
    
    
    