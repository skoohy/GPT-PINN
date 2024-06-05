import torch
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class grad_descent(object): 
    def __init__(self, nu, xt_resid, IC_xt, BC_xt, IC_u, P_resid_values, 
                 P_IC_values, P_BC_values, P_x_term, Pt_nu_P_xx_term, lr_gpt):
        # PDE parameters
        self.nu = nu
        
        # Data sizes
        self.N_R  = xt_resid.shape[0]
        self.N_IC = IC_xt.shape[0]
        self.N_BC = BC_xt.shape[0]
        
        # Precomputed / Data terms
        self.P_resid_values    = P_resid_values
        self.P_BC_values       = P_BC_values
        self.P_IC_values       = P_IC_values
        self.IC_u              = IC_u
        self.P_x_term          = P_x_term
        self.Pt_nu_P_xx_term   = Pt_nu_P_xx_term
    
        # Optimizer data/parameter
        self.lr_gpt = lr_gpt
    
    def grad_loss(self, c):
        c = c.to(device)
        #######################################################################
        #######################################################################        
        #########################  Residual Gradient  #########################

        ux   = torch.matmul(self.P_x_term, c[:,None])
        u    = torch.matmul(self.P_resid_values, c[:,None])
        u_ux = torch.mul(u, ux)
        
        ut_vuxx       = torch.matmul(self.Pt_nu_P_xx_term, c[:,None])
        first_product = torch.add(ut_vuxx, u_ux)

        u_Px    = torch.mul(u, self.P_x_term)
        P_ux    = torch.mul(self.P_resid_values, ux)
        term1      = torch.add(u_Px, P_ux)
        
        second_product = torch.add(term1, self.Pt_nu_P_xx_term)
        
        grad_list = torch.mul(2/self.N_R, torch.sum(torch.mul(first_product,second_product), axis=0))
        #######################################################################
        #######################################################################        
        ###################  Boundary and Initial Gradient  ###################   
        
        BC_term = torch.matmul(self.P_BC_values, c[:,None])
        IC_term1 = torch.matmul(self.P_IC_values, c[:,None])
        IC_term1 = torch.sub(IC_term1, self.IC_u)
        
        grad_list[:c.shape[0]] += torch.mul(2/self.N_BC, torch.sum(torch.mul(BC_term,  self.P_BC_values), axis=0))
        grad_list[:c.shape[0]] += torch.mul(2/self.N_IC, torch.sum(torch.mul(IC_term1, self.P_IC_values), axis=0))

        return grad_list

    def update(self, c):
        c = torch.sub(c, torch.mul(self.lr_gpt, self.grad_loss(c)))
        return c.expand(1,c.shape[0])