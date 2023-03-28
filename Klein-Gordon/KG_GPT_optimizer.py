import torch
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class grad_descent(object): 
    def __init__(self, alpha, beta, gamma, xt_resid, IC_xt, BC_xt, IC_u1, IC_u2,
                 BC_u, xcos_x2cos2_term, Ptt_aPxx_bP_term, alpha_P_xx_term, beta_P_term, 
                 gamm2_P_term, P_resid_values, P_IC_values, P_BC_values, 
                 Pi_t_term, lr_gpt, network_gradients):
        
        # PDE parameters
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        
        # Data sizes
        self.N_R  = xt_resid.shape[0]
        self.N_IC = IC_xt.shape[0]
        self.N_BC = BC_xt.shape[0]
        
        # Pre-computed terms
        self.network_gradients = network_gradients
        self.P_resid_values    = P_resid_values
        self.P_BC_values       = P_BC_values
        self.P_IC_values       = P_IC_values
        self.Pi_t_term         = Pi_t_term
        self.xcos_x2cos2_term  = xcos_x2cos2_term.to(device)
        self.Ptt_aPxx_bP_term   = Ptt_aPxx_bP_term
        self.alpha_P_xx_term   = alpha_P_xx_term
        self.beta_P_term       = beta_P_term
        self.gamm2_P_term      = gamm2_P_term

        # Training Solutions
        self.IC_u1 = IC_u1
        self.IC_u2 = IC_u2
        self.BC_u  = BC_u
        
        # Learning Rate
        self.lr_gpt = lr_gpt

    def grad_loss(self, c):
        c = c.to(device)
        grad_list = torch.ones(c.shape[0]).to(device)
        #######################################################################
        #######################################################################        
        #########################  Residual Gradient  #########################
        term1 = 0
        term2 = 0
        term5 = 0

        for i in range(c.shape[0]):
            P_tt = self.network_gradients[i][1]
            a_P_xx = self.alpha_P_xx_term[i]
            b_P = self.beta_P_term[i]

            term1 += c[i]*P_tt + c[i]*a_P_xx + c[i]*b_P
            term2 += c[i]*self.P_resid_values[:,i][:,None]
            
        first_product = (term1 + self.gamma*(term2**2) + self.xcos_x2cos2_term)
                
        for m in range(c.shape[0]):
            term5 = term2*self.gamm2_P_term[m]
            second_product = self.Ptt_aPxx_bP_term[m] + term5
            
            gradient = (2/self.N_R)*torch.sum(first_product*second_product)
            grad_list[m] = gradient.item()

        #######################################################################
        #######################################################################        
        ##################  Boundary and Initial 1 Gradient  ##################
        BC_term  = 0
        IC1_term = 0 
                
        for i in range(c.shape[0]):
            BC_term += c[i]*self.P_BC_values[:,i][:,None]
            IC1_term += c[i]*self.P_IC_values[:,i][:,None]
        BC_term  -= self.BC_u
        IC1_term -= self.IC_u1
        
        for m in range(c.shape[0]):
            grad_list[m] += (2/self.N_BC)*torch.sum(BC_term*self.P_BC_values[:,m][:,None])
            grad_list[m] += (2/self.N_IC)*torch.sum(IC1_term*self.P_IC_values[:,m][:,None])

        #######################################################################
        #######################################################################        
        ########################  Initial 2 Gradient  ######################### 
        IC2_term = 0
        
        for i in range(c.shape[0]):
            IC2_term += c[i]*self.Pi_t_term[:,i][:,None]
                
        for m in range(c.shape[0]):
            grad_list[m] += (2/self.N_IC)*torch.sum(IC2_term*self.Pi_t_term[:,m][:,None])
            
        return grad_list
    
    def update(self, c):
        c = c - self.lr_gpt*self.grad_loss(c)
        return c.expand(1,c.shape[0])
    
           
            