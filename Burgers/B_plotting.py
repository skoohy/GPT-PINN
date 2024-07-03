import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({'figure.max_open_warning': 0})
plt.style.use(['science', 'notebook'])

path = "./b_data/"

generation_time  = np.loadtxt(path+"generation_time.dat")
max_losses       = np.loadtxt(path+"max_losses.dat")
neurons          = np.loadtxt(path+"neurons.dat")
total_time       = np.loadtxt(path+"total_time.dat")

xt_resid         = np.loadtxt(path+"xt_resid.dat")
b_test           = np.loadtxt(path+"b_test.dat")
xt_test          = np.loadtxt(path+"xt_test.dat")

gpt_test_losses  = np.loadtxt(path+"gpt_test_losses.dat")
gpt_test_soln    = np.loadtxt(path+"gpt_test_soln.dat")
gpt_test_time    = np.loadtxt(path+"gpt_test_time.dat")
 
pinn_test_losses = np.loadtxt(path+"pinn_test_losses.dat")
pinn_test_soln   = np.loadtxt(path+"pinn_test_soln.dat")
pinn_test_time   = np.loadtxt(path+"pinn_test_time.dat")

###############################################################################
# Neurons

fig, ax = plt.subplots()
xmin = 0.005; xmax = 1
y = 1; height = 0.15

ax.hlines(y, xmin, xmax, colors="k", lw=5)
ax.vlines(xmin, y - height/4, y + height/4, colors="k", lw=5)
ax.vlines(xmax, y - height/4, y + height/4, colors="k", lw=5)

xx = neurons
yy = [y]*len(neurons)

ax.plot(xx, yy, "ko", ms=10, mfc="r")

offsetX=mpl.transforms.ScaledTranslation(.1,  0, fig.dpi_scale_trans)
offsetY=mpl.transforms.ScaledTranslation( 0, .1, fig.dpi_scale_trans)

for i, param in enumerate(neurons):
    ax.text(param-0.003, y+0.095, f"{i+1}", ha="left", va="top", 
            transform=ax.transData+offsetX-offsetY, fontsize=20)

ax.text(-0.01, y, "0.005", horizontalalignment="right", 
        verticalalignment="center", fontsize=20)
ax.text(  1.01, y,   "1.0", horizontalalignment="left", 
        verticalalignment="center", fontsize=20)
ax.text((0.005+1.0)/2, y-0.1, r"$\nu$", fontsize=25, 
        horizontalalignment="center")
ax.set_xlim(-0.1,1.1)
ax.set_ylim(0,2)
ax.axis("off")
plt.show()

###############################################################################
# Largest loss plot

fig, ax = plt.subplots()
x = range(1,len(max_losses)+1)
ax.plot(x, max_losses, "-o", color="black")
ax.set_xticks(ticks=x)
ax.set_xlim(min(x),max(x))
ax.set_yscale("log")
ax.set_xlabel("Number of neurons")
ax.set_ylabel("Largest Loss")
ax.grid(True)
plt.show()

###############################################################################
# Generation times

fig, ax = plt.subplots()
x = range(1,len(generation_time)+1)
ax.plot(x, generation_time, "-o", color="black")
ax.set_xticks(ticks=x)
ax.set_xlim(min(x),max(x))
ax.set_xlabel("Number of neurons")
ax.set_ylabel("Generation Time (minutes)")
ax.grid(True)
plt.show()

###############################################################################
# Total time

avg_time_gpt = []
for i in range(len(gpt_test_time)-1):
    avg_time_gpt.append(gpt_test_time[i+1]-gpt_test_time[i])

avg_time_pinn = []
for i in range(len(gpt_test_time)-1):
    avg_time_pinn.append(pinn_test_time[i+1]-pinn_test_time[i])
    
ratio = np.average(avg_time_gpt) / np.average(avg_time_pinn)

fig, ax = plt.subplots()
x = range(1,len(b_test)+1) 
ax.plot(x, gpt_test_time,  color="black", label="GPT-PINN", lw=3.5)
ax.plot(x, pinn_test_time, color="red", label="PINN",     lw=3.5)
ax.set_xticks(ticks=[1,5,10,15,20,25])
ax.set_xlim(min(x),max(x))
ax.set_xlabel("Test Case")
ax.set_ylim(0,max(pinn_test_time))
ax.set_ylabel("Time (hours)")
props = dict(boxstyle="round", facecolor="white", alpha=0.75, lw=2)
legend = ax.legend(frameon=True, fontsize=22, facecolor="white", 
                   framealpha=0.75)
ax.text(0.035, 0.725, f"Slope Ratio: {round(ratio,4)}", transform=ax.transAxes, 
        fontsize=17.5, verticalalignment='top', bbox=props)
ax.grid(True)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor("k") 
plt.show()

###############################################################################
# Absolute Errors

# Plot multiple in a row
start = 1
end   = 5

# Plot all
#start = 1 
#end   = len(b_test)

tests = b_test[start-1:end-1]

for i, param in enumerate(tests):
    i += start-1
    nu = param
    u_gpt  = gpt_test_soln[:,i][:,None]
    u_pinn = pinn_test_soln[:,i][:,None]
    
    L2_error = np.linalg.norm(u_gpt-u_pinn, ord=2) /\
               np.linalg.norm(u_pinn, ord=2)
    
    shape = (int(np.sqrt(u_gpt.shape[0])), int(np.sqrt(u_gpt.shape[0])))
    
    x      = xt_test[:,0].reshape(shape).transpose(1,0)
    t      = xt_test[:,1].reshape(shape).transpose(1,0)
    u_gpt  = u_gpt.reshape(shape).transpose(1,0)
    u_pinn = u_pinn.reshape(shape).transpose(1,0)
        
    fig, ax = plt.subplots()
    plot = ax.contourf(t, x, abs(u_gpt-u_pinn), 100, cmap="rainbow")
    cbar = fig.colorbar(plot)
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax.text(0.05, 0.95, f"L2 Error: {round(L2_error,4)}", 
            transform=ax.transAxes, fontsize=15, verticalalignment="top", 
            bbox=props)
    ax.set_title(fr"Error GPT-PINN: $\nu={round(nu,4)}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    plt.show()
    
    fig, ax = plt.subplots()
    plot = ax.contourf(t, x, u_gpt, 100, cmap="rainbow")
    cbar = fig.colorbar(plot)
    ax.set_title(fr"GPT-PINN: $\nu={round(nu,4)}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    plt.show()
    
    fig, ax = plt.subplots()
    plot = ax.contourf(t, x, u_pinn, 100, cmap="rainbow")
    cbar = fig.colorbar(plot)
    ax.set_title(fr"PINN: $\nu={round(nu,4)}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    plt.show()
