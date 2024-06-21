import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['science', 'notebook'])

def KG_plot(xt, u, scale=150, cmap="rainbow", title=None, 
                 dpi=150, figsize=(10,8)):
    """Klein-Gordon Contour Plot"""
    
    shape = [int(np.sqrt(u.shape[0])), int(np.sqrt(u.shape[0]))]
    
    x = xt[:,0].reshape(shape=shape).transpose(1,0).cpu().detach() 
    t = xt[:,1].reshape(shape=shape).transpose(1,0).cpu().detach() 
    u =       u.reshape(shape=shape).transpose(1,0).cpu().detach()
    
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    cp = ax.contourf(t, x, u, scale, cmap=cmap)
    fig.colorbar(cp)
        
    ax.set_xlabel("$t$", fontsize=25)
    ax.set_ylabel("$x$", fontsize=25)
        
    ax.set_xticks([ 1.0, 2.0, 3.0, 4.0, 5.0])
    ax.set_yticks([-1.0, -0.5,  0.0,  0.5, 1.0])
    
    ax.tick_params(axis='both', which='major', labelsize=22.5)
    ax.tick_params(axis='both', which='minor', labelsize=22.5)
    
    if title is not None:
        ax.set_title(title, fontsize=20)
        
    plt.show()

def loss_plot(epochs, losses, title=None, dpi=150, figsize=(10,8)):
    """Training losses"""
    plt.figure(dpi=dpi, figsize=figsize)
    plt.plot(epochs, losses, c="k", linewidth=3)
    
    plt.xlabel("Epoch",     fontsize=22.5)
    plt.ylabel("PINN Loss", fontsize=22.5)
     
    plt.grid(True)
    plt.xlim(0,max(epochs))
    plt.yscale('log')
    
    if title is not None:
        plt.title(title, fontsize=20)
    
    plt.show()