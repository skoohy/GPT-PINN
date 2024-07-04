import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({'figure.max_open_warning': 0})
plt.style.use(['science', 'notebook'])

path1 = "./ac_data/"
path2 = "./sa_ac_data/"
                                     
generation_time  = np.loadtxt(path1+"generation_time.dat")
loss_list        = np.loadtxt(path1+"loss_list.dat")
neurons          = np.loadtxt(path1+"neurons.dat")
total_time       = np.loadtxt(path1+"total_time.dat")

xt_resid         = np.loadtxt(path1+"xt_resid.dat")
ac_test          = np.loadtxt(path1+"ac_test.dat")
xt_test          = np.loadtxt(path1+"xt_test.dat")

test_gpt_losses  = np.loadtxt(path1+"test_gpt_losses.dat")
test_gpt_soln    = np.loadtxt(path1+"test_gpt_soln.dat")
test_gpt_time    = np.loadtxt(path1+"test_gpt_time.dat")
 
test_pinn_losses = np.loadtxt(path2+"test_pinn_losses.dat")
test_pinn_soln   = np.loadtxt(path2+"test_pinn_soln.dat")
test_pinn_time   = np.loadtxt(path2+"test_pinn_time.dat")

###############################################################################
# Neurons

lmbda = neurons[:,0]
eps   = neurons[:,1]

fig, ax = plt.subplots()
ax.scatter(lmbda, eps, c="k", s=60)

for i, _ in enumerate(neurons):
    ax.text(lmbda[i]+1e-6,eps[i]+.05, f"{i+1}", c="k", fontsize=20)

lmbda_ticks = np.linspace(1e-4, 1e-3,5)
eps_ticks   = np.linspace(1,5,5)

ax.set_xticks(ticks=lmbda_ticks, labels=[str(round(i,7)) for i in lmbda_ticks])
ax.set_yticks(ticks=eps_ticks,   labels=[str(i) for i in eps_ticks])

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\epsilon$")
ax.grid(True)
ax.set_axisbelow(True)
plt.show()

###############################################################################
# Largest loss plot

fig, ax = plt.subplots()
x = range(1,len(loss_list)+1)
ax.plot(x, loss_list, "-o", color="black")
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
for i in range(len(test_gpt_time)-1):
    avg_time_gpt.append(test_gpt_time[i+1]-test_gpt_time[i])

avg_time_pinn = []
for i in range(len(test_gpt_time)-1):
    avg_time_pinn.append(test_pinn_time[i+1]-test_pinn_time[i])
    
ratio = np.average(avg_time_gpt) / np.average(avg_time_pinn)

fig, ax = plt.subplots()
x = range(1,len(ac_test)+1) 
ax.plot(x, test_gpt_time,  color="black", label="GPT-PINN", lw=3.5)
ax.plot(x, test_pinn_time, color="red", label="PINN",     lw=3.5)
ax.set_xticks(ticks=[1,5,10,15,20,25])
ax.set_xlim(min(x),max(x))
ax.set_xlabel("Test Case")
ax.set_ylim(0,max(test_pinn_time))
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
end   = 3

# Plot all
#start = 1 
#end   = len(ac_test)

tests = ac_test[start-1:end-1]

for i, param in enumerate(tests):
    i += start-1
    lmbda, eps = param
    u_gpt  = test_gpt_soln[:,i][:,None]
    u_pinn = test_pinn_soln[:,i][:,None]
    
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
    ax.set_title(fr"Error GPT-PINN: $\lambda={round(lmbda,7)},$ $\epsilon={round(eps,2)}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    plt.show()
    
    fig, ax = plt.subplots()
    plot = ax.contourf(t, x, u_gpt, 100, cmap="rainbow")
    cbar = fig.colorbar(plot)
    ax.set_title(fr"GPT-PINN: $\lambda={round(lmbda,7)},$ $\epsilon={round(eps,2)}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    plt.show()
    
    fig, ax = plt.subplots()
    plot = ax.contourf(t, x, u_pinn, 100, cmap="rainbow")
    cbar = fig.colorbar(plot)
    ax.set_title(fr"PINN: $\lambda={round(lmbda,7)},$ $\epsilon={round(eps,2)}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    plt.show()