import torch

class grad_descent(): 
    def __init__(self, nu, xt_size, IC_size, BC_size, IC_u, out, out_IC,
                 out_BC, out_x, Pt_nu_P_xx_term, lr_gpt):
        # PDE parameters
        self.nu = nu
        
        # Derivative Factors
        self.fac1 = 2/xt_size
        self.fac2 = 2/BC_size
        self.fac3 = 2/IC_size
        
        # Pre-computed terms
        self.out             = out
        self.out_BC          = out_BC
        self.out_IC          = out_IC
        self.out_x           = out_x
        self.Pt_nu_P_xx_term = Pt_nu_P_xx_term
        
        # Training Solutions
        self.IC_u = IC_u
        # BC = 0

        # Learning Rate
        self.lr_gpt = lr_gpt
    
    def update(self, c, c_reshaped):
        #######################################################################
        #########################  Residual Gradient  #########################
        u    = torch.matmul(self.out, c_reshaped)
        ux   = torch.matmul(self.out_x, c_reshaped)
        u_ux = torch.mul(u, ux)
        
        ut_vuxx       = torch.matmul(self.Pt_nu_P_xx_term, c_reshaped)
        first_product = torch.add(ut_vuxx, u_ux)

        u_Px  = torch.mul(u, self.out_x)
        P_ux  = torch.mul(self.out, ux)
        term1 = torch.add(u_Px, P_ux)
        
        second_product = torch.add(term1, self.Pt_nu_P_xx_term)
        
        grad_list = torch.mul(self.fac1, torch.sum(torch.mul(first_product,
                                                   second_product), axis=0))
        #######################################################################
        ###################  Boundary and Initial Gradient  ###################   
        
        BC_term  = torch.matmul(self.out_BC, c_reshaped)
        IC_term = torch.sub(torch.matmul(self.out_IC, c_reshaped), self.IC_u)
        
        grad_list += torch.mul(self.fac2, torch.sum(torch.mul(BC_term, 
                                                    self.out_BC), axis=0))
        grad_list += torch.mul(self.fac3, torch.sum(torch.mul(IC_term, 
                                                    self.out_IC), axis=0))

        c = torch.sub(c, torch.mul(self.lr_gpt, grad_list))
        return c