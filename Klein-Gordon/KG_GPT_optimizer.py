import torch
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class grad_descent(object): 
    def __init__(self, alpha, beta, gamma, xt_resid, IC_xt, BC_xt, IC_u1, IC_u2,
                 BC_u, xcos_x2cos2_term, Ptt_aPxx_bP_term, alpha_P_xx_term, beta_P_term, 
                 gamm2_P_term, P_resid_values, P_IC_values, P_BC_values, 
                 Pi_t_term, P_xx_term, P_tt_term, epochs_gpt, lr_gpt):
        # PDE parameters
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        
        # Data sizes
        self.N_R  = xt_resid.shape[0]
        self.N_IC = IC_xt.shape[0]
        self.N_BC = BC_xt.shape[0]
        
        # Pre-computed terms
        self.P_resid_values    = P_resid_values
        self.P_BC_values       = P_BC_values
        self.P_IC_values       = P_IC_values
        self.Pi_t_term         = Pi_t_term
        self.xcos_x2cos2_term  = xcos_x2cos2_term.to(device)
        self.Ptt_aPxx_bP_term  = Ptt_aPxx_bP_term
        self.alpha_P_xx_term   = alpha_P_xx_term
        self.beta_P_term       = beta_P_term
        self.gamm2_P_term      = gamm2_P_term
        self.P_xx_term         = P_xx_term
        self.P_tt_term         = P_tt_term

        # Training Solutions
        self.IC_u1 = IC_u1
        self.IC_u2 = IC_u2
        self.BC_u  = BC_u
        
        # Learning Rate
        self.lr_gpt = lr_gpt

    def grad_loss(self, c):
        c = c.to(device)

        #######################################################################
        #######################################################################        
        #########################  Residual Gradient  #########################

        term = torch.add(torch.add(self.P_tt_term, self.alpha_P_xx_term), self.beta_P_term)
        term1 = torch.matmul(term, c[:,None])
        
        term2 = torch.matmul(self.P_resid_values, c[:,None])
            
        first_product = torch.add(torch.add(term1, torch.mul(self.gamma, torch.square(term2))), self.xcos_x2cos2_term)
        
        term3 = torch.mul(term2, self.gamm2_P_term)
        
        second_product = torch.add(self.Ptt_aPxx_bP_term, term3)
        
        grad_list = torch.mul(2/self.N_R, torch.sum(torch.mul(first_product,second_product), axis=0))

        #######################################################################
        #######################################################################        
        ##################  Boundary and Initial 1 Gradient  ##################
        
        BC_term  = torch.matmul(self.P_BC_values, c[:,None])
        IC1_term = torch.matmul(self.P_IC_values, c[:,None])
        
        BC_term  = torch.sub(BC_term,  self.BC_u)
        IC1_term = torch.sub(IC1_term, self.IC_u1)
    
        grad_list[:c.shape[0]] += torch.mul(2/self.N_BC, torch.sum(torch.mul(BC_term,  self.P_BC_values), axis=0))
        grad_list[:c.shape[0]] += torch.mul(2/self.N_IC, torch.sum(torch.mul(IC1_term, self.P_IC_values), axis=0))

        #######################################################################
        #######################################################################        
        ########################  Initial 2 Gradient  #########################         
        IC2_term = torch.matmul(self.Pi_t_term, c[:,None])
        
        grad_list[:c.shape[0]] += torch.mul(2/self.N_IC, torch.sum(torch.mul(IC2_term, self.Pi_t_term), axis=0))
        
        return grad_list
    
    def update(self, c):
        c = torch.sub(c, torch.mul(self.lr_gpt, self.grad_loss(c)))
        return c.expand(1,c.shape[0])
    
           
            