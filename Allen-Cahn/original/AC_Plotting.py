import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['science', 'notebook'])

def AC_plot(t_test, x_test, u_test, title, cmap="rainbow", scale=150):  
    """Allen-Cahn Contour Plot"""

    shape = [int(np.sqrt(u_test.shape[0])), int(np.sqrt(u_test.shape[0]))]
    
    x = x_test.reshape(shape)
    t = t_test.reshape(shape)
    u = u_test.reshape(shape)
        
    fig, ax = plt.subplots(dpi=150, figsize=(10,8))
    cp = ax.contourf(t, x, u, scale, cmap=cmap)
    cbar = fig.colorbar(cp)
    
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel("$t$", fontsize=25)
    ax.set_ylabel("$x$", fontsize=25)
    ax.set_xticks(ticks=[0.0, 0.25, 0.5, 0.75, 1.0], labels=[0.0, 0.25, 0.5, 0.75, 1.0], fontsize=18)
    ax.set_yticks(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], labels=[-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=18)
    ax.set_title(title, fontsize=20)
    
    plt.show()

def loss_plot(epochs_adam_sa, epochs_lbfgs_sa, adam_loss, lbfgs_loss, title=None, dpi=150, figsize=(10,8)):
    """Training losses"""

    x_adam  = range(0,epochs_adam_sa+250,250)
    x_lbfgs = range(x_adam[-1]+5,epochs_adam_sa+epochs_lbfgs_sa+5,5)
        
    plt.figure(dpi=dpi, figsize=figsize)
    
    plt.vlines(x_adam[-1], lbfgs_loss[0], adam_loss[-1], linewidth=3, colors='r')
    
    plt.plot(x_adam,  adam_loss, c="k", linewidth=3, label="ADAM")
    
    plt.plot(x_lbfgs, lbfgs_loss, linewidth=3, c='r', label="L-BFGS")
    
    plt.xlabel("Epoch",     fontsize=22.5)
    plt.ylabel("SA-PINN Loss", fontsize=22.5)
    plt.grid(True)
    plt.xlim(0,epochs_adam_sa+epochs_lbfgs_sa)
    plt.yscale('log')
    
    if title is not None:
        plt.title(title)
    
    plt.show()