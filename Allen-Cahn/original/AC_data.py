from torch import cos, linspace, meshgrid, hstack, zeros, vstack, pi

def initial_u(x, t=0):
    return x**2*cos(pi*x)

def boundary_min(t, x=-1):
    return -1 + 0*t

def boundary_max(t, x=1):
    return -1 + 0*t

def create_ICBC_data(Xi, Xf, Ti, Tf, BC_pts, IC_pts):
    ##########################################################
    x_BC = linspace(Xi, Xf, BC_pts)
    t_BC = linspace(Ti, Tf, BC_pts)
    X_BC, T_BC = meshgrid(x_BC, t_BC, indexing='ij')
    
    x_IC = linspace(Xi, Xf, IC_pts)
    t_IC = linspace(Ti, Ti, IC_pts)
    X_IC, T_IC = meshgrid(x_IC, t_IC, indexing='ij')
    ##########################################################
    IC_x = X_IC[:,0][:,None]
    IC_t = zeros(IC_x.shape[0], 1)
    IC_u = initial_u(IC_x)     
    IC   = hstack((IC_x, IC_t))
    ##########################################################
    BC_bottom_x = X_BC[0,:][:,None] 
    BC_bottom_t = T_BC[0,:][:,None] 
    BC_bottom_u = boundary_min(BC_bottom_t)
    BC_bottom   = hstack((BC_bottom_x, BC_bottom_t)) 
    ##########################################################
    BC_top_x = X_BC[-1,:][:,None] 
    BC_top_t = T_BC[-1,:][:,None] 
    BC_top_u = boundary_max(BC_top_t)
    BC_top   = hstack((BC_top_x, BC_top_t))
    ##########################################################
    xt_train_BC = vstack((BC_top, BC_bottom))
    u_train_BC  = vstack((BC_top_u, BC_bottom_u))
    ##########################################################
    return (IC, IC_u, xt_train_BC, u_train_BC)  

def create_residual_data(Xi, Xf, Ti, Tf, Nc, N_test):
    ##########################################################
    x_resid = linspace(Xi, Xf, Nc)
    t_resid = linspace(Ti, Tf, Nc)
    
    XX_resid, TT_resid = meshgrid((x_resid, t_resid), indexing='ij')
    
    X_resid = XX_resid.transpose(1,0).flatten()[:,None]
    T_resid = TT_resid.transpose(1,0).flatten()[:,None]
    
    xt_resid    = hstack((X_resid, T_resid))
    f_hat_train = zeros((xt_resid.shape[0], 1))
    ##########################################################
    x_test = linspace(Xi, Xf, N_test)
    t_test = linspace(Ti, Tf, N_test)
    
    XX_test, TT_test = meshgrid((x_test, t_test), indexing='ij')
    
    X_test = XX_test.transpose(1,0).flatten()[:,None]
    T_test = TT_test.transpose(1,0).flatten()[:,None]
    
    xt_test    = hstack((X_test, T_test))
    ##########################################################
    return (xt_resid, f_hat_train, xt_test)
