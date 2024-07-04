import torch

class grad_descent(): 
    def __init__(self, lmbda, eps, xt_size, IC_size, BC_size, IC_u, 
                 Pt_lPxx_eP_term, out, out_IC, out_BC_ub, out_BC_lb, 
                 out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x, 
                 epochs_gpt, lr_gpt):
        # PDE parameters
        self.lmbda = lmbda
        self.eps   = eps
                
        # Derivative Factors
        self.fac1 = 2/xt_size
        self.fac2 = 2/BC_size
        self.fac3 = 2/IC_size
        
        # Pre-computed terms
        self.out             = out
        self.out_BC_ub       = out_BC_ub
        self.out_BC_lb       = out_BC_lb
        self.out_BC_diff     = out_BC_diff
        
        self.out_BC_ub_x     = out_BC_ub_x
        self.out_BC_lb_x     = out_BC_lb_x
        self.out_BC_diff_x   = out_BC_diff_x
        
        self.out_IC          = out_IC
        self.Pt_lPxx_eP_term = Pt_lPxx_eP_term
        
        # Training Solutions
        self.IC_u = IC_u

        # Learning Rate
        self.lr_gpt = lr_gpt

    def update(self, c, c_reshaped):
        #######################################################################        
        #########################  Residual Gradient  #########################
        u = torch.matmul(self.out, c_reshaped)

        ut_luxx_eu    = torch.matmul(self.Pt_lPxx_eP_term, c_reshaped)
        eu3           = torch.mul(self.eps, torch.pow(u,3))
        first_product = torch.add(ut_luxx_eu, eu3)

        term1 = torch.mul(3*self.eps, torch.mul(torch.square(u), self.out))
        second_product = torch.add(self.Pt_lPxx_eP_term, term1)
        
        grad_list = torch.mul(self.fac1, torch.sum(torch.mul(first_product, second_product), axis=0))
        #######################################################################
        ###################  Boundary and Initial Gradient  ###################       
        IC_term = torch.matmul(self.out_IC, c_reshaped)
        IC_term = torch.sub(IC_term, self.IC_u)
        grad_list += torch.mul(self.fac3, torch.sum(torch.mul(IC_term, self.out_IC), axis=0))

        BC_term_ub = torch.matmul(self.out_BC_ub, c_reshaped)
        BC_term_lb = torch.matmul(self.out_BC_lb, c_reshaped)
        BC_term    = torch.sub(BC_term_ub, BC_term_lb)
        grad_list += torch.mul(self.fac2, torch.sum(torch.mul(BC_term, self.out_BC_diff), axis=0))
        
        BC_term_ub_x = torch.matmul(self.out_BC_ub_x, c_reshaped)
        BC_term_lb_x = torch.matmul(self.out_BC_lb_x, c_reshaped)
        BC_term_x    = torch.sub(BC_term_ub_x, BC_term_lb_x)
        grad_list   += torch.mul(self.fac2, torch.sum(torch.mul(BC_term_x, self.out_BC_diff_x), axis=0))
        
        c = torch.sub(c, torch.mul(self.lr_gpt, grad_list))
        return c