import torch

class grad_descent(): 
    def __init__(self, alpha, beta, gamma, xt_size, IC_size, BC_size, IC_u1, 
                 IC_u2, BC_u, xcos_x2cos2, Ptt_aPxx_bP, gamma2_P, out, out_IC, 
                 out_IC_t, out_BC, epochs_gpt, lr_gpt):
        # PDE parameters
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        # Derivative Factors
        self.fac1 = 2/xt_size
        self.fac2 = 2/BC_size
        self.fac3 = 2/IC_size

        # Pre-computed terms
        self.out         = out
        self.out_BC      = out_BC
        self.out_IC      = out_IC
        self.out_IC_t    = out_IC_t
        self.xcos_x2cos2 = xcos_x2cos2
        self.Ptt_aPxx_bP = Ptt_aPxx_bP
        self.gamma2_P    = gamma2_P

        # Training Solutions
        self.IC_u1 = IC_u1
        self.IC_u2 = IC_u2
        self.BC_u  = BC_u

        # Learning Rate
        self.lr_gpt = lr_gpt

    def update(self, c, c_reshaped):
        #######################################################################
        #########################  Residual Gradient  #########################
        u = torch.matmul(self.out, c_reshaped)
        
        term1 = torch.matmul(self.Ptt_aPxx_bP, c_reshaped)
        term2 = torch.add(term1, torch.mul(self.gamma, torch.square(u)))
        first_product = torch.add(term2, self.xcos_x2cos2)
        
        term3 = torch.mul(self.gamma2_P, u)
        second_product = torch.add(self.Ptt_aPxx_bP, term3)
       
        grad_list = torch.mul(self.fac1, torch.sum(torch.mul(first_product, 
                                                   second_product), axis=0))  
        #######################################################################
        ##################  Boundary and Initial 1 Gradient  ##################
        BC_term  = torch.matmul(self.out_BC, c_reshaped)
        IC1_term = torch.matmul(self.out_IC, c_reshaped)
        
        BC_term  = torch.sub(BC_term,  self.BC_u)
        IC1_term = torch.sub(IC1_term, self.IC_u1)
    
        grad_list += torch.mul(self.fac2, torch.sum(torch.mul(BC_term,  
                                                    self.out_BC), axis=0))
        grad_list += torch.mul(self.fac3, torch.sum(torch.mul(IC1_term, 
                                                    self.out_IC), axis=0))
        #######################################################################
        ########################  Initial 2 Gradient  #########################         
        IC2_term = torch.matmul(self.out_IC_t, c_reshaped)
        
        grad_list += torch.mul(self.fac3, torch.sum(torch.mul(IC2_term, 
                                                    self.out_IC_t), axis=0))

        c = torch.sub(c, torch.mul(self.lr_gpt, grad_list))
        return c