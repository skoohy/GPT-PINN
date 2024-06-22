# Third-party and Standard Libraries
from datetime import datetime
import numpy as np
import torch
import time
import os

# Modules
from KG_test import gpt_test, gpt_test_loss, pinn_test, pinn_test_loss
from KG_train import pinn_train, offline_generation
from KG_data import residual_data, ICBC_data
from KG_precomp import xcos_term, inputs
from KG_models import NN

data_dir = "./kg_data"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda")
print_seperator = 60*"*"

print(f"Start: {datetime.now()}\n")

###############################################################################
#### Domain and Simulated Data ####
Xi, Xf         = -1.0, 1.0
Ti, Tf         =  0.0, 5.0
Nc, N_test     =  100,  40
BC_pts, IC_pts =  512, 512

xt_resid, f_hat, xt_test = residual_data(Xi, Xf, Ti, Tf, Nc, N_test)
xt_resid = xt_resid.to(device)
f_hat    = f_hat.to(device)
xt_test  = xt_test.to(device) 

IC_xt, IC_u1, IC_u2, BC_xt, BC_u = ICBC_data(Xi, Xf, Ti, Tf, BC_pts, IC_pts) 
IC_xt = IC_xt.to(device)
IC_u1 = IC_u1.to(device)
IC_u2 = IC_u2.to(device)
BC_xt = BC_xt.to(device)
BC_u  = BC_u.to(device)    

#### Training Parameter Set ####
alpha    = np.linspace(-2, -1, 10)
beta     = np.linspace( 0,  1, 10)
gamma    = np.linspace( 0,  1, 10) 
kg_train = np.array(np.meshgrid(alpha, beta, gamma)).T.reshape(-1,3)

#### Forcing function ####
xcos_x2cos2 = xcos_term(xt_resid[:,0].unsqueeze(1), 
                        xt_resid[:,1].unsqueeze(1))

#### PINN Attributes ####
layers_pinn = np.array([2, 40, 40, 1])
lr_pinn     = 0.0005
epochs_pinn = 75000

#### GPT-PINN Attributes ####
number_of_neurons = 15
lr_gpt            = 0.025
epochs_gpt_train  = 2000

test_cases      = np.ceil(0.2*len(kg_train)).astype(int)
epochs_gpt_test = 5000

loss_list  = np.zeros(number_of_neurons)
neurons    = np.zeros((number_of_neurons,3))
neurons[0] = (np.median(alpha), np.median(beta), np.median(gamma))
#neurond[0] = kg_train[np.random.randint(low=0, high=len(kg_train))]

c_init = np.zeros(number_of_neurons, dtype=object)
for i in range(number_of_neurons):
    c_init[i] = torch.full((1,i+1), 1/(i+1)).to(device)

#### Data sizes ####
test_size = xt_test.shape[0]
xt_size   = xt_resid.shape[0]
IC_size   = IC_xt.shape[0]
BC_size   = BC_xt.shape[0]

#### Neuron outputs on the full training grid ####
xt_resid    = xt_resid.requires_grad_()
IC_xt       = IC_xt.requires_grad_()
out_full    = torch.ones((xt_size, number_of_neurons)).to(device)
out_BC      = torch.ones((BC_size, number_of_neurons)).to(device)
out_IC      = torch.ones((IC_size, number_of_neurons)).to(device)
out_IC_t    = torch.ones((IC_size, number_of_neurons)).to(device)
out_xx_full = torch.ones((xt_size, number_of_neurons)).to(device)
out_tt_full = torch.ones((xt_size, number_of_neurons)).to(device)

#### Neuron outputs on the test grid ####
out_test    = torch.zeros((test_size, number_of_neurons)).to(device) 

generation_time = np.zeros(number_of_neurons)

###############################################################################
total_time_1 = time.time()
for i, neuron in enumerate(neurons):
    print(print_seperator)
    kg_train = np.delete(kg_train, np.where(np.all(kg_train == neuron, 
                                                   axis=1))[0], axis=0)
    ###########################################################################
    # Full PINN to be used as activation function    
    alpha, beta, gamma = neuron
    t1 = time.time()
    PINN = NN(layers_pinn, alpha, beta, gamma, xcos_x2cos2).to(device)
    pinn_losses = pinn_train(PINN, alpha, beta, gamma, xt_resid, IC_xt, IC_u1, 
                             IC_u2, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn)
    t2 = time.time()
    print(f"PINN time: {(t2-t1)/60} minutes\n")
    ###########################################################################    
    # GPT-PINN Training / Offline Generation
    c_initial  = c_init[i][0]

    train_out, train_out_xx, train_out_tt, train_out_IC_t, train_out_IC, \
    train_out_BC = inputs(PINN, xt_resid, out_full, out_xx_full, out_tt_full, 
                          out_IC_t, out_IC, out_BC, IC_xt, BC_xt, i, 
                          out_test, xt_test)
        
    t1 = time.time()
    largest_loss, largest_case = offline_generation(kg_train, c_initial, 
    xt_size, IC_size, BC_size, IC_u1, IC_u2, BC_u, xcos_x2cos2, train_out, 
    train_out_xx, train_out_tt, train_out_IC, train_out_IC_t, train_out_BC, 
    f_hat, epochs_gpt_train, lr_gpt)
    t2 = time.time()
    generation_time[i] = (t2-t1)/60
    print(f"Generation time: {(t2-t1)/60} minutes") 
    
    ###########################################################################
    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        neurons[i+1] = largest_case
        
    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"Parameter Case: {largest_case}\n")
    
total_time = (time.time() - total_time_1) / 3600

print(print_seperator)
print(f"Total Training Time: {total_time} Hours\n")
print(f"Activation Function Parameters: \n{neurons}\n")
print(f"Loss list: {loss_list}")

###############################################################################
#### Testing ####
# Recording losses affects the overall time so a seperate function is used
# but they can easily be combined into one.

kg_test = kg_train[np.random.choice(len(kg_train), test_cases, replace=False)]
c_initial = c_init[-1][0]

print("GPT-PINN Testing Started")
gpt_test_time, gpt_test_soln = gpt_test(kg_test, out_full, out_xx_full, 
out_tt_full, out_IC, out_IC_t, out_BC, f_hat, xcos_x2cos2, xt_size, IC_size, 
BC_size, IC_u1, IC_u2, BC_u, c_initial, epochs_gpt_test, lr_gpt, out_test)

gpt_test_losses = gpt_test_loss(kg_test, out_full, out_xx_full, out_tt_full, 
out_IC, out_IC_t, out_BC, f_hat, xcos_x2cos2, xt_size, IC_size, BC_size, IC_u1, 
IC_u2, BC_u, c_initial, epochs_gpt_test, lr_gpt)
print("GPT-PINN Testing Ended\n")


print("PINN Testing Started")
pinn_test_time, pinn_test_soln = pinn_test(kg_test, layers_pinn, xcos_x2cos2, 
xt_resid, IC_xt, IC_u1, IC_u2, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, 
xt_test)

pinn_test_losses = pinn_test_loss(kg_test, layers_pinn, xcos_x2cos2, xt_resid, 
IC_xt, IC_u1, IC_u2, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn)
print("PINN Testing Ended")

np.savetxt(data_dir+"/generation_time.dat", generation_time)
np.savetxt(data_dir+"/loss_list.dat",       loss_list)
np.savetxt(data_dir+"/neurons.dat",         neurons)
np.savetxt(data_dir+"/total_time.dat",      np.array([total_time]))

np.savetxt(data_dir+"/xt_resid.dat",        xt_resid.detach().cpu().numpy())
np.savetxt(data_dir+"/kg_test.dat",         kg_test)
np.savetxt(data_dir+"/xt_test.dat",         xt_test.cpu().numpy())

np.savetxt(data_dir+"/gpt_test_losses.dat", gpt_test_losses)
np.savetxt(data_dir+"/gpt_test_soln.dat",   gpt_test_soln)
np.savetxt(data_dir+"/gpt_test_time.dat",   gpt_test_time+total_time)

np.savetxt(data_dir+"/pinn_test_losses.dat", pinn_test_losses)
np.savetxt(data_dir+"/pinn_test_soln.dat",   pinn_test_soln)
np.savetxt(data_dir+"/pinn_test_time.dat",   pinn_test_time)

params = {"Device":device,
          "Domain": {"Xi": Xi, "Xf": Xf, "Ti":Ti, "Tf":Tf}, 
          "Data sizes": {"Nc":Nc, "N_test":N_test, "BC_pts":BC_pts, "IC_pts":IC_pts},
          "layers_pinn":layers_pinn,
          "lr_pinn":lr_pinn,
          "epochs_pinn":epochs_pinn,
          "parameter size":len(kg_train)+number_of_neurons,
          "number_of_neurons":number_of_neurons,
          "lr_gpt":lr_gpt,
          "epochs_gpt_train":epochs_gpt_train,
          "test_cases":test_cases,
          "epochs_gpt_test":epochs_gpt_test}

np.save(data_dir+"/params.npy", params)

print(f"\nEnd: {datetime.now()}")
