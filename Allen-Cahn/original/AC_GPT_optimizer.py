import torch
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class grad_descent(object): 
    def __init__(self, lmbda, eps, 
                 xt_resid, IC_xt, BC_xt, IC_u, BC_u,
                 P_resid_values, P_IC_values, P_BC_values, Pt_lPxx_eP_term, lr_gpt):
        # PDE parameters
        self.lmbda = lmbda
        self.eps   = eps
                
        # Data sizes
        self.N_R  = xt_resid.shape[0]
        self.N_IC = IC_xt.shape[0]
        self.N_BC = BC_xt.shape[0]
        
        # Pre-computed terms
        self.P_resid_values = P_resid_values
        self.P_BC_values = P_BC_values
        self.P_IC_values = P_IC_values
        self.Pt_lPxx_eP_term = Pt_lPxx_eP_term
        
        # Training data
        self.IC_u = IC_u
        self.BC_u = BC_u

        # Optimizer data/parameter
        self.lr_gpt = lr_gpt

    def grad_loss(self, c):
        c = c.to(device)
        #######################################################################
        #######################################################################        
        #########################  Residual Gradient  #########################
        
        ut_luxx_eu = torch.matmul(self.Pt_lPxx_eP_term, c[:,None])
        u = torch.matmul(self.P_resid_values, c[:,None])
        eu3 = torch.mul(self.eps, torch.pow(u,3))
        first_product = torch.add(ut_luxx_eu, eu3)

        term1 = torch.mul(3*self.eps, torch.mul(torch.square(u), self.P_resid_values))
        second_product1 = torch.add(self.Pt_lPxx_eP_term, term1)
        grad_list = torch.mul(2/self.N_R, torch.sum(torch.mul(first_product, second_product1), axis=0))
        
        #######################################################################
        #######################################################################        
        ###################  Boundary and Initial Gradient  ###################       
        
        BC_term = torch.matmul(self.P_BC_values, c[:,None])
        BC_term = torch.sub(BC_term, self.BC_u)
                
        IC_term = torch.matmul(self.P_IC_values, c[:,None])
        IC_term = torch.sub(IC_term, self.IC_u)
        
        grad_list[:c.shape[0]] += torch.mul(2/self.N_BC, torch.sum(torch.mul(BC_term, self.P_BC_values), axis=0))
        grad_list[:c.shape[0]] += torch.mul(2/self.N_IC, torch.sum(torch.mul(IC_term, self.P_IC_values), axis=0))

        return grad_list

    def update(self, c):
        c = torch.sub(c, torch.mul(self.lr_gpt, self.grad_loss(c)))
        return c.expand(1,c.shape[0])