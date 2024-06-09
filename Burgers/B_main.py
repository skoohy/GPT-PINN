# The program is set up to generate N neurons for the GPT-PINN. Once N neurons 
# has been achieved it is further trained to examine the largest loss 
# over all parameters once more. This is not needed for practical use.
# Set "train_final = False" (Line 60), if you wish to remove this behavior.

# Third-party and Standard Libraries
from datetime import datetime
import numpy as np
import torch
import time
import os

print(f"Program Start: {datetime.now()}\n")

# Modules
from B_test import gpt_test, gpt_test_loss, pinn_test, pinn_test_loss
from B_data import residual_data, ICBC_data
from B_train import offline_generation
from B_train import pinn_train
from B_precomp import inputs
from B_models import NN

data_dir = "./b_data/"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda")
print_seperator = 60*"*"

###############################################################################
#### Domain and Simulated Data ####
Xi, Xf         = -1.0, 1.0
Ti, Tf         =  0.0, 1.0
Nc, N_test     =  100, 100
BC_pts, IC_pts =  200, 200

xt_resid, f_hat, xt_test = residual_data(Xi, Xf, Ti, Tf, Nc, N_test)
xt_resid = xt_resid.to(device)
f_hat    = f_hat.to(device)
xt_test  = xt_test.to(device) 

IC_xt, IC_u, BC_xt, BC_u = ICBC_data(Xi, Xf, Ti, Tf, BC_pts, IC_pts) 
IC_xt = IC_xt.to(device)
IC_u  = IC_u.to(device)
BC_xt = BC_xt.to(device)
BC_u  = BC_u.to(device) 

#### Training Parameter Set ####
b_train = np.linspace(0.005, 1, 129)

#### PINN Attributes ####
layers_pinn = np.array([2, 20, 20, 20, 20, 1])
lr_pinn     = 0.005
epochs_pinn = 60000
tol         = 2e-5

#### GPT-PINN Attributes ####
train_final       = True
number_of_neurons = 9
lr_gpt            = 0.02
epochs_gpt_train  = 2000
neurons           = np.zeros(number_of_neurons)
neurons[0]        = np.median(b_train)
#neurons[0]        = b_train[np.random.randint(low=0, high=len(b_train))]

#### GPT-PINN Test Attributes ####
test_cases      = 25
epochs_gpt_test = 5000

#### Data sizes ####
test_size = xt_test.shape[0]
xt_size   = xt_resid.shape[0]
IC_size   = IC_xt.shape[0]
BC_size   = BC_xt.shape[0]

#### Neuron outputs on the full training grid ####
xt_resid    = xt_resid.requires_grad_()
out_full    = torch.zeros((xt_size, number_of_neurons)).to(device)
out_BC      = torch.zeros((BC_size, number_of_neurons)).to(device)
out_IC      = torch.zeros((IC_size, number_of_neurons)).to(device)
out_t_full  = torch.zeros((xt_size, number_of_neurons)).to(device)
out_x_full  = torch.zeros((xt_size, number_of_neurons)).to(device)
out_xx_full = torch.zeros((xt_size, number_of_neurons)).to(device)

#### Neuron outputs on the test grid ####
out_test = torch.zeros((test_size, number_of_neurons)).to(device) 

num_largest_mag = int(xt_size*0.2)
idx_list        = torch.zeros((number_of_neurons, num_largest_mag),
                              dtype=torch.long)
loss_list       = np.zeros(number_of_neurons)
generation_time = np.zeros(number_of_neurons)

print("GPT-PINN Training Started")
total_time_1 = time.time()
for i, neuron in enumerate(neurons):
    print(print_seperator)
    # No need to train over parameters already used as neurons
    b_train = np.delete(b_train, np.where(b_train == neuron)[0])
    
    ###########################################################################
    # Full PINN to be used as activation function
    nu = neuron
    
    t1 = time.time()
    PINN = NN(layers_pinn, nu).to(device)
    pinn_losses = pinn_train(PINN, nu, xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                             f_hat, epochs_pinn, lr_pinn, tol)
    t2 = time.time()
    print(f"PINN time: {(t2-t1)/60} minutes\n")
    ###########################################################################    
    # GPT-PINN Training / Offline Generation
    train_out, train_out_x, train_out_t, train_out_xx, train_out_IC, \
    train_out_BC = inputs(PINN, xt_resid, out_full, out_t_full, out_x_full, 
                          out_xx_full, out_IC, out_BC, IC_xt, BC_xt, i, 
                          out_test, xt_test, xt_size, num_largest_mag, 
                          idx_list)
    
    if (train_final == False) and (i+1 == number_of_neurons):
        end = number_of_neurons-1
        break
    
    t1 = time.time()
    largest_loss, largest_case = offline_generation(b_train, xt_size, IC_size, 
    BC_size, IC_u, BC_u, train_out, train_out_x, train_out_t, train_out_xx, 
    train_out_IC, train_out_BC, f_hat, epochs_gpt_train, lr_gpt, neurons, i)
    t2 = time.time()
    generation_time[i] = (t2-t1)/60
    print(f"Generation time: {(t2-t1)/60} minutes") 
    ###########################################################################
    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        neurons[i+1] = largest_case

    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"Parameter Case: {largest_case}\n")
        
    if (i+1 == number_of_neurons):
        end = number_of_neurons
        break

total_time = (time.time() - total_time_1) / 3600      

print(print_seperator)
print("GPT-PINN Training Ended\n")
print(f"Total training time: {total_time} Hours\n")
print(f"Activation function parameters: \n{neurons}\n")
print(f"Largest loss list: \n{loss_list[:end]}\n")

###############################################################################
#### Testing ####
# Recording losses affects the overall time so a seperate function is used
# but they can easily be combined into one.

b_test = b_train[np.random.choice(len(b_train), test_cases, replace=False)]

print("GPT-PINN Testing Started")
gpt_test_time, gpt_test_soln = gpt_test(b_test, xt_size, IC_size, BC_size, 
IC_u, BC_u, train_out, train_out_x, train_out_t, train_out_xx, train_out_IC, 
train_out_BC, f_hat, epochs_gpt_test, lr_gpt, neurons, out_test)

gpt_test_losses = gpt_test_loss(b_test, xt_size, IC_size, BC_size, IC_u, BC_u, 
train_out, train_out_x, train_out_t, train_out_xx, train_out_IC, train_out_BC, 
f_hat, epochs_gpt_test, lr_gpt, neurons)
print("GPT-PINN Testing Ended\n")

print("PINN Testing Started")
pinn_test_time, pinn_test_soln = pinn_test(b_test, layers_pinn, xt_resid, 
IC_xt, IC_u, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, xt_test, tol)

pinn_test_losses = pinn_test_loss(b_test, layers_pinn, xt_resid, IC_xt, IC_u, 
BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, tol)
print("PINN Testing Ended\n")

np.savetxt(data_dir+"/generation_time.dat",  generation_time[:end])
np.savetxt(data_dir+"/max_losses.dat",       loss_list[:end])
np.savetxt(data_dir+"/neurons.dat",          neurons)
np.savetxt(data_dir+"/total_time.dat",       np.array([total_time]))

np.savetxt(data_dir+"/xt_resid.dat",         xt_resid.detach().cpu().numpy())
np.savetxt(data_dir+"/b_test.dat",           b_test)
np.savetxt(data_dir+"/xt_test.dat",          xt_test.cpu().numpy())

np.savetxt(data_dir+"/gpt_test_losses.dat",  gpt_test_losses)
np.savetxt(data_dir+"/gpt_test_soln.dat",    gpt_test_soln)
np.savetxt(data_dir+"/gpt_test_time.dat",    gpt_test_time+total_time)

np.savetxt(data_dir+"/pinn_test_losses.dat", pinn_test_losses)
np.savetxt(data_dir+"/pinn_test_soln.dat",   pinn_test_soln)
np.savetxt(data_dir+"/pinn_test_time.dat",   pinn_test_time)

params = {"Device":device,
          "Domain": {"Xi": Xi, "Xf": Xf, "Ti":Ti, "Tf":Tf}, 
          "Data sizes": {"Nc":Nc, "N_test":N_test, "BC_pts":BC_pts, "IC_pts":IC_pts},
          "tol":tol,
          "layers_pinn":layers_pinn,
          "lr_pinn":lr_pinn,
          "epochs_pinn":epochs_pinn,
          "parameter size":len(b_train)+number_of_neurons,
          "number_of_neurons":number_of_neurons,
          "lr_gpt":lr_gpt,
          "epochs_gpt_train":epochs_gpt_train,
          "test_cases":test_cases,
          "epochs_gpt_test":epochs_gpt_test,
          "num_largest_mag":num_largest_mag}

np.save(data_dir+"/params.npy", params)

print(f"Program End: {datetime.now()}\n")
