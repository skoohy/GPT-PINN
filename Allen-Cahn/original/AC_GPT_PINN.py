import torch
import torch.nn as nn

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
class GPT(nn.Module):
    def __init__(self, layers, lmbda, eps, P, c_initial, xt_resid, IC_xt, 
                 BC_xt, IC_u, BC_u, f_hat, activation_resid, activation_IC, 
                 activation_BC, Pt_lPxx_eP_term):
        super().__init__()
        self.layers = layers
        
        self.lmbda = lmbda
        self.eps   = eps
        
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1], bias=False) for i in range(len(layers)-1)])
        self.activation = P
        
        self.activation_IC     = activation_IC
        self.activation_BC     = activation_BC
        self.activation_resid  = activation_resid
        self.Pt_lPxx_eP_term   = Pt_lPxx_eP_term

        self.IC_u     = IC_u
        self.BC_u     = BC_u
        self.f_hat    = f_hat
        self.xt_resid = xt_resid
        self.IC_xt    = IC_xt
        self.BC_xt    = BC_xt
        
        self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[1].weight.data = c_initial
        
    def forward(self, datatype=None, test_data=None):
        if test_data is not None:
            a = torch.Tensor().to(device)
            for i in range(0, self.layers[1]):
                a = torch.cat((a, self.activation[i](test_data)), 1)
            final_output = self.linears[-1](a)
            
            return final_output
        
        if datatype == 'residual':
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
        u = self.forward(datatype='residual')

        ut_luxx_eu = torch.matmul(self.Pt_lPxx_eP_term, self.linears[1].weight.data[0][:,None])
        eu3 = torch.mul(self.eps, torch.pow(u,3))
        f = torch.add(ut_luxx_eu, eu3)
        
        return self.loss_function(f, self.f_hat)
    
    def lossICBC(self, datatype):
        """Initial and both boundary condition loss function"""
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