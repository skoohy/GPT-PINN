# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time

from B_data import create_residual_data, create_ICBC_data
from B_Plotting import Burgers_plot, loss_plot 

# Full PINN
from B_PINN import NN
from B_PINN_train import pinn_train

# Burgers GPT-PINN
from B_GPT_activation import P
from B_GPT_precomp import autograd_calculations, Pt_nu_P_xx
from B_GPT_PINN import GPT
from B_GPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")
if torch.cuda.is_available():
    print(f"Current Device Name: {torch.cuda.get_device_name()}")

torch.manual_seed(1234)
np.random.seed(1234)

# Domain and Data
Xi, Xf         = -1.0, 1.0
Ti, Tf         =  0.0, 1.0
Nc, N_test     =  100, 100
BC_pts, IC_pts =  200, 200 

residual_data = create_residual_data(Xi, Xf, Ti, Tf, Nc, N_test)
xt_resid      = residual_data[0].to(device)
f_hat         = residual_data[1].to(device)
xt_test       = residual_data[2].to(device) 

ICBC_data = create_ICBC_data(Xi, Xf, Ti, Tf, BC_pts, IC_pts)
IC_xt     = ICBC_data[0].to(device)
IC_u      = ICBC_data[1].to(device)
BC_xt     = ICBC_data[2].to(device)
BC_u      = ICBC_data[3].to(device)

# Training Parameter Set
nu_training = np.loadtxt("nu_training.txt")

train_final_gpt   = True
number_of_neurons = 9
loss_list         = np.ones(number_of_neurons)
print(f"Expected Final GPT-PINN Depth: {[2,number_of_neurons,1]}\n")

###############################################################################
#################################### Setup ####################################
###############################################################################

P_resid_values = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_IC_values    = torch.ones((   IC_xt.shape[0], number_of_neurons)).to(device)
P_BC_values    = torch.ones((   BC_xt.shape[0], number_of_neurons)).to(device)

P_t_term  = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_x_term  = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_xx_term = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)

nu_neurons    = [1 for i in range(number_of_neurons)] # Neuron parameters
nu_neurons[0] = 0.5025

num_largest_mag  = int(xt_resid.shape[0]*0.2)
idx_list         = torch.ones((number_of_neurons, num_largest_mag),dtype=torch.long)

P_list = np.ones(number_of_neurons, dtype=object)

lr_pinn     = 0.005
epochs_pinn = 60000

layers_pinn = np.array([2, 20, 20, 20, 20, 1])
tol         = 2e-5

lr_gpt          = 0.02
epochs_gpt      = 2000
epochs_gpt_test = 5000
test_cases      = 25

# Save Data/Plot Options
save_data         = False
plot_pinn_loss    = True
plot_pinn_sol     = True
plot_largest_loss = True

pinn_train_times = np.ones(number_of_neurons)
gpt_train_times  = np.ones(number_of_neurons)

total_train_time_1 = time.perf_counter()
###############################################################################
################################ Training Loop ################################
###############################################################################
for i in range(0, number_of_neurons):
    print("******************************************************************")
    ########################### Full PINN Training ############################
    nu_pinn_train = nu_neurons[i]
    nu_training = np.delete(nu_training, np.where(nu_training == nu_pinn_train)[0])

    pinn_train_time_1 = time.perf_counter()
    PINN = NN(layers_pinn, nu_pinn_train).to(device)

    if (i+1 == number_of_neurons):
        print(f"Begin Final Full PINN Training: nu={nu_pinn_train} (Obtaining Neuron {i+1})")
    else:
        print(f"Begin Full PINN Training: nu={nu_pinn_train} (Obtaining Neuron {i+1})")
        
    pinn_losses = pinn_train(PINN, nu_pinn_train, xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                             f_hat, epochs_pinn, lr_pinn, tol)

    pinn_train_time_2 = time.perf_counter()
    print(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")

    w1 = PINN.linears[0].weight.detach().cpu()
    w2 = PINN.linears[1].weight.detach().cpu()
    w3 = PINN.linears[2].weight.detach().cpu()
    w4 = PINN.linears[3].weight.detach().cpu()
    w5 = PINN.linears[4].weight.detach().cpu()
    
    b1 = PINN.linears[0].bias.detach().cpu()
    b2 = PINN.linears[1].bias.detach().cpu()
    b3 = PINN.linears[2].bias.detach().cpu()
    b4 = PINN.linears[3].bias.detach().cpu()
    b5 = PINN.linears[4].bias.detach().cpu()
        
    P_list[i] = P(w1, w2, w3, w4, w5, b1, b2, b3, b4, b5).to(device)

    print(f"\nCurrent GPT-PINN Depth: [2,{i+1},1]")
    
    if (save_data):        
        path = fr".\Full-PINN-Data (B)\({nu_pinn_train})"
        
        if not os.path.exists(path):
            os.makedirs(path)
    
        np.savetxt(fr"{path}\saved_w1.txt", w1.numpy())
        np.savetxt(fr"{path}\saved_w2.txt", w2.numpy())
        np.savetxt(fr"{path}\saved_w3.txt", w3.numpy())

        np.savetxt(fr"{path}\saved_b1.txt", b1.numpy())
        np.savetxt(fr"{path}\saved_b2.txt", b2.numpy())
        np.savetxt(fr"{path}\saved_b3.txt", b3.numpy())
        
        np.savetxt(fr"{path}\u_test.txt", PINN(xt_test).detach().cpu().numpy())
        np.savetxt(fr"{path}\losses.txt", pinn_losses[0])
        np.savetxt(fr"{path}\epochs.txt", pinn_losses[1])
        
    if (plot_pinn_sol):
        u = PINN(xt_test)
        Burgers_plot(xt_test, u, title=fr"PINN Solution $\nu={nu_pinn_train}$")
    
    if (plot_pinn_loss):
        loss_vals = pinn_losses[0]
        epochs    = pinn_losses[1]
        loss_plot(epochs, loss_vals, title=fr"PINN Losses $\nu={nu_pinn_train}$")

    if (i == number_of_neurons-1) and (train_final_gpt == False):
        break

    ############################ GPT-PINN Training ############################
    layers_gpt = np.array([2, i+1, 1])
    P_t, P_x, P_xx = autograd_calculations(xt_resid, P_list[i]) 
        
    val, index      = torch.sort(torch.abs(P_xx.view(-1)))
    largest_indices = torch.LongTensor(index[xt_resid.shape[0]-num_largest_mag:xt_resid.shape[0]].cpu())
    idx_list[i]     = largest_indices

    P_t_term[:,i][:,None]  = P_t
    P_x_term[:,i][:,None]  = P_x
    P_xx_term[:,i][:,None] = P_xx
    
    P_IC_values[:,i][:,None]    = P_list[i](IC_xt)
    P_BC_values[:,i][:,None]    = P_list[i](BC_xt)
    P_resid_values[:,i][:,None] = P_list[i](xt_resid)
    
    P_t_term[:,i][:,None].put_(idx_list[i].to(device),       torch.zeros(num_largest_mag).to(device))
    P_x_term[:,i][:,None].put_(idx_list[i].to(device),       torch.zeros(num_largest_mag).to(device))
    P_xx_term[:,i][:,None].put_(idx_list[i].to(device),      torch.zeros(num_largest_mag).to(device))
    P_resid_values[:,i][:,None].put_(idx_list[i].to(device), torch.zeros(num_largest_mag).to(device))
    
    # Finding The Next Neuron
    largest_case = 0
    largest_loss = 0

    if (i+1 == number_of_neurons):
        print("\nBegin Final GPT-PINN Training (Largest Loss Training)")
    else:
        print(f"\nBegin GPT-PINN Training (Finding Neuron {i+2} / Largest Loss Training)")

    gpt_train_time_1 = time.perf_counter()
    for nu in nu_training:
        if layers_gpt[1] == 1:
            c_initial = torch.ones(layers_gpt[1])
    
        elif nu in nu_neurons[:i+1]: 
            index     = np.where(nu == nu_neurons[:i+1]) 
            c_initial = torch.zeros(layers_gpt[1])
            c_initial[index] = 1
        
        else:
            dist = np.zeros(layers_gpt[1])    
            for k, nu_neuron in enumerate(nu_neurons[:i+1]): 
                dist[k] = np.abs(nu_neuron - nu)
    
            d      = np.argsort(dist) 
            first  = d[0] 
            second = d[1] 
    
            a = dist[first]
            b = dist[second] 
            bottom = a+b
            
            c_initial = torch.zeros(layers_gpt[1])
            c_initial[first]  = b / bottom 
            c_initial[second] = a / bottom
        c_initial = c_initial[None,:]

        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, P_t_term[:,0:i+1], P_xx_term[:,0:i+1])      

        GPT_NN = GPT(layers_gpt, nu, P_list[0:i+1], c_initial, IC_u, BC_u, 
                     f_hat, P_resid_values[:,0:i+1], P_IC_values[:,0:i+1], P_BC_values[:,0:i+1], 
                     Pt_nu_P_xx_term[:,0:i+1], P_x_term[:,0:i+1]).to(device)
    
        gpt_losses = gpt_train(GPT_NN, nu, xt_resid, IC_xt, IC_u, BC_xt, 
                               BC_u, P_resid_values[:,0:i+1], P_IC_values[:,0:i+1], 
                               P_BC_values[:,0:i+1], Pt_nu_P_xx_term[:,0:i+1], 
                               P_x_term[:,0:i+1], epochs_gpt, lr_gpt, 
                               largest_loss, largest_case)
        
        largest_loss = gpt_losses[0]
        largest_case = gpt_losses[1]
    
    gpt_train_time_2 = time.perf_counter()
    print("GPT-PINN Training Completed")
    print(f"\nGPT Training Time ({i+1} Neurons): {(gpt_train_time_2-gpt_train_time_1)/3600} Hours")
    
    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        nu_neurons[i+1] = largest_case
        
    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"Parameter Case: {largest_case}")
total_train_time_2 = time.perf_counter()                       

###############################################################################
# Results of largest loss, parameters chosen, and times may vary based on
# the initialization of full PINN and the final loss of the full PINN
print("******************************************************************")
print("*** Full PINN and GPT-PINN Training Complete ***")
print(f"Total Training Time: {(total_train_time_2-total_train_time_1)/3600} Hours\n")
print(f"Final GPT-PINN Depth: {[2,len(P_list),1]}")
print(f"\nActivation Function Parameters: \n{nu_neurons}\n")

for j in range(number_of_neurons-1):
    print(f"Largest Loss of GPT-PINN Depth {[2,j+1,2]}: {loss_list[j]}")
if (train_final_gpt):
    print(f"Largest Loss of GPT-PINN Depth {[2,j+2,2]}: {loss_list[-1]}")
        
if (plot_largest_loss):
    plt.figure(dpi=150, figsize=(10,8))
    
    if (train_final_gpt):
        range_end = number_of_neurons + 1
        list_end  = number_of_neurons
    else:
        range_end = number_of_neurons 
        list_end  = number_of_neurons - 1
        
    plt.plot(range(1,range_end), loss_list[:list_end], marker='o', markersize=7, 
             c="k", linewidth=3)
    
    plt.grid(True)
    plt.xlim(1,max(range(1,range_end)))
    plt.xticks(range(1,range_end))
    
    plt.yscale("log") 
    plt.xlabel("Number of Neurons",      fontsize=17.5)
    plt.ylabel("Largest Loss",           fontsize=17.5)
    plt.title("GPT-PINN Largest Losses", fontsize=17.5)
    plt.show()

############################### GPT-PINN Testing ############################## 
nu_test = nu_training.tolist()
for i in nu_neurons: 
    if (i in nu_test):
        nu_test.remove(i)

idx = np.random.choice(len(nu_test), test_cases, replace=False)
nu_test = np.array(nu_test)[idx]

print(f"\nBegin GPT-PINN Testing ({len(set(idx.flatten()))} Cases)")

layers_gpt = np.array([2, len(P_list), 1])

total_test_time_1 = time.perf_counter()
incremental_test_times = np.ones(len(nu_test))
cnt = 0

for nu_test_param in nu_test:
    dist = np.zeros(layers_gpt[1])    
    for k, nu_neuron in enumerate(nu_neurons): 
        dist[k] = np.abs(nu_neuron - nu_test_param)

    d      = np.argsort(dist) 
    first  = d[0] 
    second = d[1] 

    a = dist[first]
    b = dist[second] 
    bottom = a+b
        
    c_initial = torch.zeros(layers_gpt[1])
    c_initial[first]  = b / bottom 
    c_initial[second] = a / bottom
    c_initial = c_initial[None,:]
        
    Pt_nu_P_xx_term = Pt_nu_P_xx(nu_test_param, P_t_term, P_xx_term)      

    GPT_NN = GPT(layers_gpt, nu_test_param, P_list, c_initial, IC_u, BC_u, f_hat, 
                 P_resid_values, P_IC_values, P_BC_values, Pt_nu_P_xx_term, 
                 P_x_term).to(device)
    
    gpt_losses = gpt_train(GPT_NN, nu_test_param, xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                           P_resid_values, P_IC_values, P_BC_values, 
                           Pt_nu_P_xx_term, P_x_term, epochs_gpt_test, lr_gpt, 
                           largest_loss, largest_case, testing=True)
    
    incremental_test_times[cnt] = (time.perf_counter()-total_test_time_1)/3600
    cnt += 1

#np.savetxt(".\incremental_test_times.txt", incremental_test_times)

total_test_time_2 = time.perf_counter()
print("\nGPT-PINN Testing Completed")
print(f"\nTotal Testing Time: {(total_test_time_2-total_test_time_1)/3600} Hours")

init_time = (total_train_time_2-total_train_time_1)/3600
test_time = incremental_test_times
line = test_time + init_time
x = range(1,test_time.shape[0]+1)
plt.figure(dpi=150, figsize=(10,8))
plt.plot(x, line, c="k", lw=3.5)
plt.xlabel("Test Case Number", fontsize=22.5)
plt.ylabel("Time (Hours)", fontsize=22.5)
plt.xlim(min(x),max(x))
plt.ylim(min(line),max(line))
xtick = list(range(0,test_cases+1,5))
xtick[0] = 1
plt.xticks(xtick)
plt.grid(True)
plt.show()
