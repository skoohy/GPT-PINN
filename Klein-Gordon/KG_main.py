# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time

from KG_data import create_residual_data, create_ICBC_data
from KG_Plotting import KG_plot, loss_plot 

# Full PINN
from KG_PINN import NN
from KG_PINN_train import pinn_train

# GPT-PINN
from KG_GPT_activation import P
from KG_GPT_precomp import autograd_calculations, xcos_x2cos2, Ptt_aPxx_bP, gamma2_P, Pi_t
from KG_GPT_PINN import GPT
from KG_GPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device: {device}')
if torch.cuda.is_available():
    print(f'Current Device Name: {torch.cuda.get_device_name()}')
    
# Domain and Data
Xi, Xf         = -1.0, 1.0
Ti, Tf         =  0.0, 5.0
Nc, N_test     =  100,  40
BC_pts, IC_pts =  512, 512

residual_data = create_residual_data(Xi, Xf, Ti, Tf, Nc, N_test)
xt_resid      = residual_data[0].to(device)
f_hat         = residual_data[1].to(device)
xt_test       = residual_data[2].to(device) 

ICBC_data = create_ICBC_data(Xi, Xf, Ti, Tf, BC_pts, IC_pts)
IC_xt     = ICBC_data[0].to(device)
IC_u1     = ICBC_data[1].to(device)
IC_u2     = ICBC_data[2].to(device)
BC_xt     = ICBC_data[3].to(device)
BC_u      = ICBC_data[4].to(device)    

# Training Parameter Set
Alpha = np.linspace(-2, -1, 10)
Beta  = np.linspace(0, 1, 10)
Gamma = np.linspace(0, 1, 10)

kg_training = []
for i in range(Alpha.shape[0]):
    for j in range(Beta.shape[0]):
        for k in range(Gamma.shape[0]):
            kg_training.append([Alpha[i],Beta[j],Gamma[k]])
kg_training = np.array(kg_training)

train_final_gpt   = True
number_of_neurons = 15
loss_list         = np.ones(number_of_neurons) # Store largest losses
print(f"Expected Final GPT-PINN Depth: {[2,number_of_neurons,1]}\n")

###############################################################################
#################################### Setup ####################################
###############################################################################

P_resid_values = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_IC_values    = torch.ones((   IC_xt.shape[0], number_of_neurons)).to(device)
P_BC_values    = torch.ones((   BC_xt.shape[0], number_of_neurons)).to(device)
Pi_t_term      = torch.ones((   IC_xt.shape[0], number_of_neurons)).to(device)

P_xx_term = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_tt_term = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)

xcos_x2cos2_term = xcos_x2cos2(xt_resid[:,[0]], xt_resid[:,[1]]).to(device)

kg_neurons    = [1 for i in range(number_of_neurons)]
kg_neurons[0] = [-1.5, 0.5, 0.5]

P_list = np.ones(number_of_neurons, dtype=object)

lr_pinn     = 0.0005
epochs_pinn = 75000

layers_pinn = np.array([2, 40, 40, 1])

lr_gpt     = 0.025
epochs_gpt = 2000
epochs_gpt_test = 5000
test_cases = 200

# Save Data/Plot Options
save_data         = False
plot_pinn_loss    = True
plot_pinn_sol     = True
plot_largest_loss = True

pinn_train_times = np.ones(number_of_neurons)
gpt_train_times = np.ones(number_of_neurons)

total_train_time_1 = time.perf_counter()
###############################################################################
################################ Training Loop ################################
###############################################################################
for i in range(0, number_of_neurons):
    print("******************************************************************")
    ########################### Full PINN Training ############################
    kg_pinn_train = kg_neurons[i]
    alpha, beta, gamma = kg_pinn_train[0], kg_pinn_train[1], kg_pinn_train[2]
    
    pinn_train_time_1 = time.perf_counter()
    PINN = NN(layers_pinn, alpha, beta, gamma, xcos_x2cos2_term).to(device)
    
    if (i+1 == number_of_neurons):
        print(f"Begin Final Full PINN Training: alpha={alpha}, beta={beta}, gamma={gamma} (Obtaining Neuron {i+1})")
    else:
        print(f"Begin Full PINN Training: alpha={alpha}, beta={beta}, gamma={gamma} (Obtaining Neuron {i+1})")
        
    pinn_losses = pinn_train(PINN, alpha, beta, gamma, xt_resid, IC_xt, IC_u1, IC_u2, 
                             BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn)
    
    pinn_train_time_2 = time.perf_counter()
    print(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
    
    w1 = PINN.linears[0].weight.detach().cpu()
    w2 = PINN.linears[1].weight.detach().cpu()
    w3 = PINN.linears[2].weight.detach().cpu()
    
    b1 = PINN.linears[0].bias.detach().cpu()
    b2 = PINN.linears[1].bias.detach().cpu()
    b3 = PINN.linears[2].bias.detach().cpu()
    
    P_list[i] = P(w1, w2, w3, b1, b2, b3).to(device) # Add new activation functions
    
    print(f"\nCurrent GPT-PINN Depth: [2,{i+1},1]")
    
    if (save_data):        
        path = fr".\Full-PINN-Data (KG)\({kg_pinn_train})"
        
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
        KG_plot(xt_test, u, title=fr"PINN Solution $\alpha={round(alpha,3)}, \beta={round(beta,3)}, \gamma={round(gamma,3)}$")
    
    if (plot_pinn_loss):
        loss_vals = pinn_losses[0]
        epochs    = pinn_losses[1]
        loss_plot(epochs, loss_vals, title=fr"PINN Losses $\alpha={round(alpha,3)}, \beta={round(beta,3)}, \gamma={round(gamma,3)}$")

    if (i == number_of_neurons-1) and (train_final_gpt == False):
        break
        
    ############################ GPT-PINN Training ############################    
    layers_gpt = np.array([2, i+1, 1])
    c_initial  = torch.full((1,i+1), 1/(i+1))
    P_xx, P_tt = autograd_calculations(xt_resid, P_list[i])  
    
    P_xx_term[:,i][:,None] = P_xx
    P_tt_term[:,i][:,None] = P_tt
    
    P_IC_values[:,i][:,None]    = P_list[i](IC_xt)
    P_BC_values[:,i][:,None]    = P_list[i](BC_xt)
    P_resid_values[:,i][:,None] = P_list[i](xt_resid)
    Pi_t_term[:,i][:,None]      = Pi_t(IC_xt, P_list[i])
    
    largest_case = 0
    largest_loss = 0
    
    if (i+1 == number_of_neurons):
        print(f"\nBegin Final GPT-PINN Training (Finding Neuron {i+2} / Using {i+1} Neurons)")
    else:
        print(f"\nBegin GPT-PINN Training (Finding Neuron {i+2} / Using {i+1} Neurons)")
        
    gpt_train_time_1 = time.perf_counter()
    for kg_param in kg_training[:200]:
        alpha, beta, gamma = kg_param[0], kg_param[1], kg_param[2]
        
        Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, P_tt_term[:,0:i+1], P_xx_term[:,0:i+1], P_resid_values[:,0:i+1])
        gamm2_P_term     = gamma2_P(gamma, P_resid_values[:,0:i+1])
        
        GPT_NN = GPT(layers_gpt, alpha, beta, gamma, P_list[0:i+1], c_initial,
                     IC_u1, IC_u2, BC_u, f_hat, xcos_x2cos2_term,
                     P_resid_values[:,0:i+1], P_IC_values[:,0:i+1], P_BC_values[:,0:i+1],
                     Pi_t_term[:,0:i+1], Ptt_aPxx_bP_term[:,0:i+1]).to(device)

        gpt_losses = gpt_train(GPT_NN, alpha, beta, gamma, xt_resid, IC_xt, IC_u1, IC_u2, BC_xt, BC_u,
                               xcos_x2cos2_term, Ptt_aPxx_bP_term[:,0:i+1], gamm2_P_term[:,0:i+1], 
                               P_resid_values[:,0:i+1], P_IC_values[:,0:i+1], P_BC_values[:,0:i+1], 
                               Pi_t_term[:,0:i+1], epochs_gpt, lr_gpt, largest_loss, largest_case)
    
        largest_loss = gpt_losses[0]
        largest_case = gpt_losses[1]
    
    gpt_train_time_2 = time.perf_counter()
    print("GPT-PINN Training Completed")
    print(f"GPT Training Time ({i+1} Neurons): {(gpt_train_time_2-gpt_train_time_1)/3600} Hours\n")
    
    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        kg_neurons[i+1] = largest_case
        
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
print(f"\nActivation Function Parameters: \n{kg_neurons}\n")

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
kg_test = kg_training.tolist()
for i in kg_neurons: 
    if (i in kg_test):
        kg_test.remove(i)

idx = np.random.choice(len(kg_test), test_cases, replace=False)
kg_test = np.array(kg_test)[idx]

print(f"\nBegin GPT-PINN Testing ({len(set(idx.flatten()))} Cases)")

I = len(P_list)   
layers_gpt = np.array([2, I, 1])
c_initial  = torch.full((1,I), 1/(I))

total_test_time_1 = time.perf_counter()
#incremental_test_times = np.ones(len(kg_test))
cnt = 0

for kg_test_param in kg_test:
    alpha, beta, gamma = kg_test_param[0], kg_test_param[1], kg_test_param[2]
    
    Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, P_tt_term, P_xx_term, P_resid_values)
    gamm2_P_term     = gamma2_P(gamma, P_resid_values)
    
    GPT_NN = GPT(layers_gpt, alpha, beta, gamma, P_list, c_initial,
                 IC_u1, IC_u2, BC_u, f_hat, xcos_x2cos2_term,
                 P_resid_values, P_IC_values, P_BC_values,
                 Pi_t_term, Ptt_aPxx_bP_term).to(device)

    gpt_losses = gpt_train(GPT_NN, alpha, beta, gamma, xt_resid, IC_xt, IC_u1, 
                           IC_u2, BC_xt, BC_u, xcos_x2cos2_term, Ptt_aPxx_bP_term, 
                           gamm2_P_term, P_resid_values, P_IC_values, P_BC_values, 
                           Pi_t_term, epochs_gpt, lr_gpt, testing=True)
    
    #incremental_test_times[cnt] = (time.perf_counter()-total_test_time_1)/3600
    #cnt += 1

#np.savetxt(".\incremental_test_times.txt", incremental_test_times)

total_test_time_2 = time.perf_counter()
print("\nGPT-PINN Testing Completed")
print(f"\nTotal Testing Time: {(total_test_time_2-total_test_time_1)/3600} Hours")

