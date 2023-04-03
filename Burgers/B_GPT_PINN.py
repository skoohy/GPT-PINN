import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPT(nn.Module):
    def __init__(self, layers, nu, P, initial_c, IC_u, BC_u, f_hat,
                 activation_resid, activation_IC, activation_BC, 
                 Pt_nu_P_xx_term, P_x_term):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1], bias=False) for i in range(len(layers)-1)])
        self.activation = P
        
        self.activation_resid = activation_resid
        self.activation_IC    = activation_IC
        self.activation_BC    = activation_BC
        
        self.Pt_nu_P_xx_term = Pt_nu_P_xx_term
        self.P_x_term        = P_x_term

        self.IC_u     = IC_u
        self.BC_u     = BC_u
        self.f_hat    = f_hat
        
        self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[1].weight.data = initial_c
        
    def forward(self, datatype=None, test_data=None):
        if test_data is not None: # Test Data Forward Pass
            a = torch.Tensor().to(device)
            for i in range(0, self.layers[1]):
                a = torch.cat((a, self.activation[i](test_data)), 1)
            final_output = self.linears[-1](a)
            return final_output
        
        if datatype == 'residual': # Residual Data Output
            final_output = self.linears[-1](self.activation_resid).to(device)
            return final_output
        
        if datatype == 'initial':
            final_output = self.linears[-1](self.activation_IC).to(device)
            return final_output
        
        if datatype == 'boundary':
            final_output = self.linears[-1](self.activation_BC).to(device)
            return final_output
    
    def lossR(self):
        """Residual loss function"""
        u  = self.forward(datatype='residual')
        ux = torch.matmul(self.P_x_term, self.linears[1].weight.data[0][:,None])
        u_ux = torch.mul(u, ux)
        ut_vuxx = torch.matmul(self.Pt_nu_P_xx_term, self.linears[1].weight.data[0][:,None])
        f = torch.add(ut_vuxx, u_ux)
        return self.loss_function(f, self.f_hat)
    
    def lossICBC(self, datatype):
        """First initial and both boundary condition loss function"""
        if datatype=='initial':
            return self.loss_function(self.forward(datatype), self.IC_u)
            
        elif datatype=='boundary':
            return self.loss_function(self.forward(datatype), self.BC_u)
 
    def loss(self):
        """Total Loss Function"""
        loss_R   = self.lossR()
        loss_IC = self.lossICBC(datatype='initial')
        loss_BC  = self.lossICBC(datatype='boundary')
        return loss_R + loss_IC + loss_BC 