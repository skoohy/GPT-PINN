# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
import scipy.io
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs
np.random.seed(1234) # lhs

# GPT-PINN
from AC_Plotting import AC_plot, loss_plot 

from AC_GPT_activation import P
from AC_GPT_precomp import autograd_calculations, Pt_lPxx_eP
from AC_GPT_PINN import GPT
from AC_GPT_train import gpt_train

# SA-PINN is implemented explicitly in the code

torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device (PyTorch): {device}")
if torch.cuda.is_available():
    print(f"Current Device Name (PyTorch): {torch.cuda.get_device_name()}")

# Domain and Data
data_path = r".\data"
xt_test = torch.from_numpy(np.loadtxt(fr"{data_path}\xt_test.txt",dtype=np.float32)).to(device)

# Initial Condition
IC_x = torch.from_numpy(np.loadtxt(fr"{data_path}\initial\x0.txt",dtype=np.float32))
IC_t = torch.zeros(IC_x.shape[0])[:,None]
val, idx = torch.sort(IC_x)
IC_u = torch.from_numpy(np.loadtxt(fr"{data_path}\initial\u0.txt",dtype=np.float32))
IC_u = IC_u[idx][:,None].to(device)
IC_xt = torch.hstack((IC_x[idx][:,None],IC_t)).to(device)

# Boundary Condition
BC_x_ub = torch.from_numpy(np.loadtxt(fr"{data_path}\boundary\x_ub.txt",dtype=np.float32))[:,None]
BC_t_ub = torch.from_numpy(np.loadtxt(fr"{data_path}\boundary\t_ub.txt",dtype=np.float32))[:,None]
BC_xt_ub = torch.hstack((BC_x_ub,BC_t_ub))

BC_x_lb = torch.from_numpy(np.loadtxt(fr"{data_path}\boundary\x_lb.txt",dtype=np.float32))[:,None]
BC_t_lb = torch.from_numpy(np.loadtxt(fr"{data_path}\boundary\t_lb.txt",dtype=np.float32))[:,None]
BC_xt_lb = torch.hstack((BC_x_lb,BC_t_lb))

BC_xt = torch.vstack((BC_xt_ub,BC_xt_lb)).to(device) 
BC_u = torch.full((200,1), -1.0).to(device)  

# Residual 
x_resid = torch.from_numpy(np.loadtxt(fr"{data_path}\f\xf_train.txt",dtype=np.float32))[:,None]
t_resid = torch.from_numpy(np.loadtxt(fr"{data_path}\f\tf_train.txt",dtype=np.float32))[:,None]

xt_resid = torch.hstack((x_resid,t_resid)).to(device) 
f_hat = torch.full((20000,1), 0.0).to(device)   

# Training Parameter Set
ac_training = np.loadtxt("ac_param_training.txt")

train_final_gpt   = True
number_of_neurons = 9
loss_list         = np.ones(number_of_neurons)
print(f"Expected Final GPT-PINN Depth: {[2,number_of_neurons,1]}\n")

###############################################################################
#################################### GPT Setup ####################################
###############################################################################
P_resid_values    = torch.ones((xt_resid.shape[0],  number_of_neurons)).to(device)
P_IC_values       = torch.ones((   IC_xt.shape[0],  number_of_neurons)).to(device)
P_BC_values       = torch.ones((   BC_xt.shape[0],  number_of_neurons)).to(device)

P_t_term  = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_xx_term = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)

ac_neurons    = [1 for i in range(number_of_neurons)]
ac_neurons[0] = [0.00055, 3.0]

P_list = np.ones(number_of_neurons, dtype=object)

lr_gpt          = 0.0025
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
#################################### SA Setup #################################
###############################################################################
# SA-PINN from https://github.com/levimcclenny/SA-PINNs

lr_adam_sa  = 0.005
lr_lbfgs_sa = 0.8

epochs_adam_sa  = 10000
epochs_lbfgs_sa = 10000

layers_sa = [2, 128, 128, 128, 128, 1]

lb = np.array([-1.0])
ub = np.array([1.0])

N0 = 512
N_b = 100
N_f = 20000

data = scipy.io.loadmat('AC.mat')
t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)

#grab training points from domain
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tf.float32)

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]
# Grab collocation points using latin hypercube sampling
X_f = lb + (ub-lb)*lhs(2, N_f)
x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)

# IC/BC data 
X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
x0 = tf.cast(X0[:,0:1], dtype = tf.float32)
t0 = tf.cast(X0[:,1:2], dtype = tf.float32)
x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)
x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)

#L-BFGS weight getting and setting from https://github.com/pierremtb/PINNs-TF2.0
def set_weights(model, w, sizes_w, sizes_b):
    with tf.device('/GPU:0'):
        for i, layer in enumerate(model.layers[0:]):
            start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
            end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(sizes_w[i] / sizes_b[i])
            weights = tf.reshape(weights, [w_div, sizes_b[i]])
            biases = w[end_weights:end_weights + sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

def get_weights(model):
    with tf.device('/GPU:0'):
        w = []
        for layer in model.layers[0:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)

        w = tf.convert_to_tensor(w)
        return w

#define the neural network model
def neural_net(layer_sizes):
    with tf.device('/GPU:0'):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
        for width in layer_sizes[1:-1]:
            model.add(layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        model.add(layers.Dense(
                layer_sizes[-1], activation=None,
                kernel_initializer="glorot_normal"))
    return model

#define the loss
def loss(x_f_batch, t_f_batch,
         x0, t0, u0, x_lb,
         t_lb, x_ub, t_ub,
         col_weights, u_weights, lmbda, eps):
    with tf.device('/GPU:0'):
        f_u_pred = f_model(x_f_batch, t_f_batch, lmbda, eps)
        u0_pred = u_model(tf.concat([x0, t0], 1))

        u_lb_pred, u_x_lb_pred, = u_x_model(u_model, x_lb, t_lb)
        u_ub_pred, u_x_ub_pred, = u_x_model(u_model, x_ub, t_ub)

        mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))
        mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred, u_ub_pred))) + \
                  tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))

        mse_f_u = tf.reduce_mean(tf.square(col_weights * f_u_pred[0]))

    return  mse_0_u + mse_b_u + mse_f_u , tf.reduce_mean(tf.square((u0 - u0_pred))), mse_b_u, tf.reduce_mean(tf.square(f_u_pred))

@tf.function
def f_model(x,t,lmbda,eps):
    with tf.device('/GPU:0'):
        u = u_model(tf.concat([x, t],1))
        u_x = tf.gradients(u, x)
        u_xx = tf.gradients(u_x, x)
        u_t = tf.gradients(u,t)
        c1 = tf.constant(lmbda, dtype = tf.float32)
        c2 = tf.constant(eps, dtype = tf.float32)
        f_u = u_t - c1*u_xx + c2*u*u*u - c2*u
    return f_u

@tf.function
def u_x_model(u_model, x, t):
    with tf.device('/GPU:0'):
        u = u_model(tf.concat([x, t],1))
        u_x = tf.gradients(u, x)
    return u, u_x

@tf.function
def grad(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, lmbda, eps):
    with tf.device('/GPU:0'):
        with tf.GradientTape(persistent=True) as tape:
            loss_value, mse_0, mse_b, mse_f = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, 
                                                   x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, lmbda, eps)
            grads = tape.gradient(loss_value, u_model.trainable_variables)
            grads_col = tape.gradient(loss_value, col_weights)
            grads_u = tape.gradient(loss_value, u_weights)
            gradients_u = tape.gradient(mse_0, u_model.trainable_variables)
            gradients_f = tape.gradient(mse_f, u_model.trainable_variables)

    return loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, gradients_u, gradients_f

def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, tf_iter, newton_iter, lmbda, eps):
    with tf.device('/GPU:0'):
        batch_sz = N_f # Can adjust batch size for collocation points, here we set it to N_f
        n_batches =  N_f // batch_sz
        adam_losses = []
        #create optimizer s for the network weights, collocation point mask, and initial boundary mask
        tf_optimizer = tf.keras.optimizers.Adam(learning_rate = lr_adam_sa, beta_1=.99)
        tf_optimizer_weights = tf.keras.optimizers.Adam(learning_rate = lr_adam_sa, beta_1=.99)
        tf_optimizer_u = tf.keras.optimizers.Adam(learning_rate = lr_adam_sa, beta_1=.99)

        print("Starting ADAM training")
        # For mini-batch (if used)
        for epoch in range(tf_iter):
            for i in range(n_batches):

                x0_batch = x0
                t0_batch = t0
                u0_batch = u0

                x_f_batch = x_f[i*batch_sz:(i*batch_sz + batch_sz),]
                t_f_batch = t_f[i*batch_sz:(i*batch_sz + batch_sz),]

                loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, g_u, g_f = grad(u_model, x_f_batch, t_f_batch, x0_batch, 
                                                                                            t0_batch,  u0_batch, x_lb, t_lb, x_ub, t_ub, 
                                                                                            col_weights, u_weights, lmbda, eps)

                tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
                tf_optimizer_weights.apply_gradients(zip([-grads_col, -grads_u], [col_weights, u_weights]))

            if (epoch % 250 == 0) or (epoch == (tf_iter-1)):
                adam_losses.append(loss_value)
                if (epoch % 1000 == 0) or (epoch == (tf_iter-1)):
                     print("Epoch: %d | " % (epoch), end='')
                     tf.print(f"mse_0: {mse_0} | mse_b: {mse_b} | mse_f: {mse_f} | Total Loss: {loss_value}\n")

        print("Starting L-BFGS training")
        loss_and_flat_grad = get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, 
                                                    x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, lmbda, eps)

        lbfgs_losses = lbfgs(loss_and_flat_grad,
          get_weights(u_model),
          Struct(), maxIter=newton_iter, learningRate=lr_lbfgs_sa)[3]
    return adam_losses, lbfgs_losses

#L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, 
                           u_weights, lmbda, eps):
    def loss_and_flat_grad(w):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as tape:
                set_weights(u_model, w, sizes_w, sizes_b)
                loss_value, _, _, _ = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, 
                                           t_ub, col_weights, u_weights, lmbda, eps)
            grad = tape.gradient(loss_value, u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

    return loss_and_flat_grad

def predict(X_star):
    with tf.device('/GPU:0'):
        X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
        u_star, _ = u_x_model(u_model, X_star[:,0:1], X_star[:,1:2])
    return u_star.numpy()

sizes_w = []
sizes_b = []
with tf.device('/GPU:0'):
    for q, width in enumerate(layers_sa):
        if q != 1:
            sizes_w.append(int(width * layers_sa[1]))
            sizes_b.append(int(width if q != 0 else layers_sa[1]))

###############################################################################
################################ Training Loop ################################
###############################################################################
for i in range(0, number_of_neurons):
    print("******************************************************************")
    ########################### SA PINN Training ############################

    ac_sa_train = ac_neurons[i]
    lmbda, eps  = float(ac_sa_train[0]), float(ac_sa_train[1])

    col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
    u_weights   = tf.Variable(100*tf.random.uniform([N0, 1]))

    #initialize the NN
    u_model = neural_net(layers_sa)

    if (i+1 == number_of_neurons):
        print(f"Begin Final SA-PINN Training: lambda={lmbda}, eps={eps} (Obtaining Neuron {i+1})")
    else:
        print(f"Begin SA-PINN Training: lambda={lmbda}, eps={eps} (Obtaining Neuron {i+1})")
    
    pinn_train_time_1 = time.perf_counter()
    
    sa_losses = fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, 
        tf_iter = epochs_adam_sa, newton_iter = epochs_lbfgs_sa, lmbda=lmbda, eps=eps)
    
    pinn_train_time_2 = time.perf_counter()
    
    print("\nSA-PINN Training Completed")
    print(f"SA-PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
    
    w1 = u_model.layers[0].get_weights()[0].T
    w2 = u_model.layers[1].get_weights()[0].T
    w3 = u_model.layers[2].get_weights()[0].T
    w4 = u_model.layers[3].get_weights()[0].T
    w5 = u_model.layers[4].get_weights()[0].T
    
    b1 = u_model.layers[0].get_weights()[1]
    b2 = u_model.layers[1].get_weights()[1]
    b3 = u_model.layers[2].get_weights()[1]
    b4 = u_model.layers[3].get_weights()[1]
    b5 = u_model.layers[4].get_weights()[1]

    P_list[i] = P(w1, w2, w3, w4, w5, b1, b2, b3, b4, b5).to(device)

    print(f"\nCurrent GPT-PINN Depth: [2,{i+1},1]")

    if (save_data):        
        path = fr".\Full-PINN-Data (AC)\({ac_sa_train})"
        
        if not os.path.exists(path):
            os.makedirs(path)
    
        np.savetxt(fr"{path}\saved_w1.txt", w1.numpy())
        np.savetxt(fr"{path}\saved_w2.txt", w2.numpy())
        np.savetxt(fr"{path}\saved_w3.txt", w3.numpy())
        np.savetxt(fr"{path}\saved_w4.txt", w3.numpy())
        np.savetxt(fr"{path}\saved_w5.txt", w3.numpy())

        np.savetxt(fr"{path}\saved_b1.txt", b1.numpy())
        np.savetxt(fr"{path}\saved_b2.txt", b2.numpy())
        np.savetxt(fr"{path}\saved_b3.txt", b3.numpy())
        np.savetxt(fr"{path}\saved_b4.txt", b3.numpy())
        np.savetxt(fr"{path}\saved_b5.txt", b3.numpy())
        
        x_test = xt_test[:,0].view(-1).cpu().detach().numpy()
        t_test = xt_test[:,1].view(-1).cpu().detach().numpy()
        X_star = np.hstack((x_test[:,None], t_test[:,None]))
        u = predict(X_star)
        
        np.savetxt(fr"{path}\u_test.txt", u)
        np.savetxt(fr"{path}\adam_losses.txt",  sa_losses[0])
        np.savetxt(fr"{path}\lbfgs_losses.txt", sa_losses[1])
        
    if (plot_pinn_sol):
        x_test = xt_test[:,0].view(-1).cpu().detach().numpy()
        t_test = xt_test[:,1].view(-1).cpu().detach().numpy()
        X_star = np.hstack((x_test[:,None], t_test[:,None]))
        u = predict(X_star)
    
        AC_plot(t_test, x_test, u, title=fr"SA-PINN Solution $\lambda={lmbda}, \epsilon={eps}$")
    
    if (plot_pinn_loss):
        adam_loss  = sa_losses[0]
        lbfgs_loss = sa_losses[1]
        loss_plot(epochs_adam_sa, epochs_lbfgs_sa, adam_loss, lbfgs_loss,
                  title=fr"SA-PINN Losses $\lambda={lmbda}, \epsilon={eps}$")

    if (i == number_of_neurons-1) and (train_final_gpt == False):
        break
        
    ############################ GPT-PINN Training ############################
    layers_gpt = np.array([2, i+1, 1])
    P_t, P_xx = autograd_calculations(xt_resid, P_list[i])

    P_t_term[:,i][:,None]  = P_t
    P_xx_term[:,i][:,None] = P_xx

    P_IC_values[:,i][:,None]    = P_list[i](IC_xt)
    P_BC_values[:,i][:,None]    = P_list[i](BC_xt)
    P_resid_values[:,i][:,None] = P_list[i](xt_resid)
    
    largest_case = 0
    largest_loss = 0
    
    if (i+1 == number_of_neurons):
        print("\nBegin Final GPT-PINN Training (Largest Loss Training)")
    else:
        print(f"\nBegin GPT-PINN Training (Finding Neuron {i+2} / Largest Loss Training)")
        
    gpt_train_time_1 = time.perf_counter()
    for ac_param in ac_training:
        lmbda, eps = ac_param[0], ac_param[1]
        
        if ([lmbda, eps] in ac_neurons):
            idx = ac_neurons.index([lmbda, eps])
                
            c_initial = torch.full((1,i+1), 0.)
            c_initial[0][idx] = 1.
        else:
            c_initial = torch.full((1,i+1), 1/(i+1))
                
        Pt_lPxx_eP_term = Pt_lPxx_eP(P_t_term[:,0:i+1], P_xx_term[:,0:i+1], P_resid_values[:,0:i+1], lmbda, eps) 
    
        GPT_NN = GPT(layers_gpt, lmbda, eps, P_list[0:i+1], c_initial, xt_resid, IC_xt, 
                     BC_xt, IC_u, BC_u, f_hat, P_resid_values[:,0:i+1], P_IC_values[:,0:i+1],
                     P_BC_values[:,0:i+1], Pt_lPxx_eP_term[:,0:i+1]).to(device)

        gpt_losses = gpt_train(GPT_NN, lmbda, eps, xt_resid, IC_xt, BC_xt, IC_u, BC_u,
                               P_resid_values[:,0:i+1], P_IC_values[:,0:i+1], 
                               P_BC_values[:,0:i+1], Pt_lPxx_eP_term[:,0:i+1],
                               lr_gpt, epochs_gpt, largest_loss, largest_case)
        
        largest_loss = gpt_losses[0]
        largest_case = gpt_losses[1]
        
        
    gpt_train_time_2 = time.perf_counter()
    loss_list[i] = largest_loss
    
    print("GPT-PINN Training Completed")
    print(f"GPT Training Time ({i+1} Neurons): {(gpt_train_time_2-gpt_train_time_1)/3600} Hours")

    if (i+1 < number_of_neurons):
        ac_neurons[i+1] = largest_case
        
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
print(f"\nActivation Function Parameters: \n{ac_neurons}\n")

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
ac_test = ac_training.tolist()
for i in ac_neurons: 
    if (i in ac_test):
        ac_test.remove(i)    
    
idx = np.random.choice(len(ac_test), test_cases, replace=False)
ac_test = np.array(ac_test)[idx]
    
print(f"\nBegin GPT-PINN Testing ({len(set(idx.flatten()))} Cases)")

I = len(P_list)   
layers_gpt = np.array([2, I, 1])
c_initial  = torch.full((1,I), 1/(I))    

total_test_time_1 = time.perf_counter()
incremental_test_times = np.ones(len(ac_test))
cnt = 0

for ac_test_param in ac_test:
    lmbda, eps = ac_test_param[0], ac_test_param[1]
    
    Pt_lPxx_eP_term = Pt_lPxx_eP(P_t_term, P_xx_term, P_resid_values, lmbda, eps) 

    GPT_NN = GPT(layers_gpt, lmbda, eps, P_list, c_initial, xt_resid, IC_xt, 
                 BC_xt, IC_u, BC_u, f_hat, P_resid_values, P_IC_values,
                 P_BC_values, Pt_lPxx_eP_term).to(device)

    gpt_losses = gpt_train(GPT_NN, lmbda, eps, xt_resid, IC_xt, BC_xt, IC_u, BC_u,
                           P_resid_values, P_IC_values, P_BC_values, Pt_lPxx_eP_term, 
                           lr_gpt, epochs_gpt, testing=True)
    
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
