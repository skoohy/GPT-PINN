# Third-party and Standard Libraries
from datetime import datetime
import tensorflow as tf
import numpy as np
import scipy.io
import torch
import time
import os

# Modules
from tensorflow.keras.models import Sequential
from AC_test import gpt_test, gpt_test_loss
from AC_train import offline_generation
from eager_lbfgs import lbfgs, Struct
from tensorflow.keras import layers
from AC_precompute import inputs
from AC_models import P
from pyDOE import lhs

"""
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
"""

seed = 0
glorot_init = tf.keras.initializers.GlorotNormal(seed=seed)
tf.keras.utils.set_random_seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda")
print_seperator = 60*"*"

data_dir = "./ac_data"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(f"Start: {datetime.now()}\n")

###############################################################################
#### Domain and simulated data ####
N0  = 512
N_b = 100
N_f = 20000

data = scipy.io.loadmat('./AC.mat')
t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact_u = data['uu']

#### SA-PINN ####
# IC 
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tf.float32)
X0 = np.concatenate((x0, 0*x0), 1)
x0 = tf.cast(X0[:,0:1], dtype = tf.float32)
t0 = tf.cast(X0[:,1:2], dtype = tf.float32)

# BC
lb = np.array([-1.0]) 
ub = np.array([1.0])

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb    = t[idx_t,:]

X_lb = np.concatenate((0*tb + lb[0], tb), 1) 
X_ub = np.concatenate((0*tb + ub[0], tb), 1) 
x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)
x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)

# Collocation
X_f = lb + (ub-lb)*lhs(2, N_f)
x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)

#### GPT ####
# IC
val, idx = torch.sort(torch.Tensor(X0[:,0:1].flatten()))
IC_u  = torch.Tensor(Exact_u[idx_x,0:1])[idx].to(device)

IC_xt = torch.hstack((torch.Tensor(X0[idx,0:1]),
                      torch.Tensor(X0[:,1:2]))).to(device)

# BC
x_lb_py = torch.Tensor(X_lb[:,0:1])
t_lb_py = torch.Tensor(X_lb[:,1:2])
x_ub_py = torch.Tensor(X_ub[:,0:1])
t_ub_py = torch.Tensor(X_ub[:,1:2])

BC_xt_ub = torch.hstack((x_ub_py,t_ub_py)).to(device).requires_grad_() 
BC_xt_lb = torch.hstack((x_lb_py,t_lb_py)).to(device).requires_grad_() 

# Collocation
xt_resid = torch.hstack((torch.Tensor(X_f[:,0:1]),
                         torch.Tensor(np.abs(X_f[:,1:2])))).to(device).requires_grad_() 
f_hat = torch.zeros(20000,1).to(device) 

x_test, t_test = torch.meshgrid((torch.linspace(-1, 1, 60),
                                 torch.linspace( 0, 1, 60)), indexing="ij")

xt_test = torch.hstack((x_test.transpose(1,0).flatten().unsqueeze(1),
                        t_test.transpose(1,0).flatten().unsqueeze(1))).to(device)

#### Training parameter set ####
lmbda        = np.linspace(1e-3,1e-4,11)
eps          = np.linspace(1,5,11)
ac_train     = np.array(np.meshgrid(lmbda, eps)).T.reshape(-1,2)
ac_train_all = ac_train.copy() 

#### SA-PINN attributes ####
lr_adam_sa  = 0.005
lr_lbfgs_sa = 0.8

epochs_adam_sa  = 10000
epochs_lbfgs_sa = 10000

layers_sa = [2, 128, 128, 128, 128, 1]

#### GPT-PINN attributes ####
number_of_neurons = 9
lr_gpt            = 0.0025
epochs_gpt_train  = 2000

test_cases      = np.ceil(0.2*len(ac_train)).astype(int)
epochs_gpt_test = 5000

loss_list  = np.zeros(number_of_neurons)
neurons    = np.zeros((number_of_neurons,2))
neurons[0] = (np.median(lmbda), np.median(eps))
#neurond[0] = ac_train[np.random.randint(low=0, high=len(ac_train))]

c_init = np.zeros(number_of_neurons, dtype=object)
for i in range(number_of_neurons):
    c_init[i] = torch.full((1,i+1), 1/(i+1)).to(device)
    
#### Data sizes ####
test_size = xt_test.shape[0]
xt_size   = xt_resid.shape[0]
IC_size   = IC_xt.shape[0]
BC_size   = BC_xt_ub.shape[0]    

#### Neuron outputs on the full training grid ####
out_full    = torch.zeros((xt_size, number_of_neurons)).to(device)
out_t_full  = torch.zeros((xt_size, number_of_neurons)).to(device)
out_xx_full = torch.zeros((xt_size, number_of_neurons)).to(device)
out_IC      = torch.zeros((IC_size, number_of_neurons)).to(device)

out_BC_ub   = torch.zeros((BC_size, number_of_neurons)).to(device)
out_BC_lb   = torch.zeros((BC_size, number_of_neurons)).to(device)
out_BC_diff = torch.zeros((BC_size, number_of_neurons)).to(device)

out_BC_ub_x   = torch.zeros((BC_size, number_of_neurons)).to(device)
out_BC_lb_x   = torch.zeros((BC_size, number_of_neurons)).to(device)
out_BC_diff_x = torch.zeros((BC_size, number_of_neurons)).to(device)

#### Neuron outputs on the full training grid ####
out_test    = torch.zeros((test_size, number_of_neurons)).to(device) 

# SA-PINN parameters
SA_w = np.zeros(len(layers_sa)-1, dtype=object)
SA_b = np.zeros(len(layers_sa)-1, dtype=object)

generation_time = np.zeros(number_of_neurons)

###############################################################################
#### SA-PINN https://github.com/levimcclenny/SA-PINNs ####

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
            model.add(layers.Dense(width, activation=tf.nn.tanh, kernel_initializer=glorot_init))
        model.add(layers.Dense(layer_sizes[-1], activation=None, kernel_initializer=glorot_init)) 
                
    return model

#define the loss
def loss(x_f_batch, t_f_batch, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, 
         u_weights, lmbda, eps):
    with tf.device('/GPU:0'):
        f_u_pred = f_model(x_f_batch, t_f_batch, lmbda, eps)
        u0_pred = u_model(tf.concat([x0, t0], 1))

        u_lb_pred, u_x_lb_pred, = u_x_model(u_model, x_lb, t_lb)
        u_ub_pred, u_x_ub_pred, = u_x_model(u_model, x_ub, t_ub)

        mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))
        mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred, u_ub_pred))) + \
                  tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))
        mse_f_u = tf.reduce_mean(tf.square(col_weights * f_u_pred[0]))
        
        mse_u = tf.reduce_mean(tf.square((u0 - u0_pred)))
        mse_f = tf.reduce_mean(tf.square(f_u_pred))
        
    return mse_0_u + mse_b_u + mse_f_u , mse_u, mse_b_u, mse_f
            
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
def grad(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, 
         x_ub, t_ub, col_weights, u_weights, lmbda, eps):
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

def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, 
        tf_iter, newton_iter, lmbda, eps, record=False):
    with tf.device('/GPU:0'):
        batch_sz = N_f # Can adjust batch size for collocation points, here we set it to N_f
        n_batches =  N_f // batch_sz

        #create optimizer s for the network weights, collocation point mask, and initial boundary mask
        tf_optimizer = tf.keras.optimizers.Adam(learning_rate = lr_adam_sa, beta_1=.99)
        tf_optimizer_weights = tf.keras.optimizers.Adam(learning_rate = lr_adam_sa, beta_1=.99)
        #tf_optimizer_u = tf.keras.optimizers.Adam(learning_rate = lr_adam_sa, beta_1=.99)

        if (record == True):
            adam_losses = np.zeros(newton_iter)

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

            
            if (epoch % 1000 == 0) or (epoch == (tf_iter-1)):
                print(f"Epoch: {epoch}")
                tf.print(f"mse_0: {mse_0} | mse_b: {mse_b} | mse_f: {mse_f} | Total Loss: {loss_value}\n")

            if (record == True):
                adam_losses[epoch] = loss_value 

        print("Starting L-BFGS training")
        loss_and_flat_grad = get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, 
                                                    x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, lmbda, eps)
        if (record == True):
            lbfgs_losses = lbfgs(loss_and_flat_grad, get_weights(u_model), Struct(), 
                                 maxIter=newton_iter, learningRate=lr_lbfgs_sa, record=True)[3]
            return adam_losses, lbfgs_losses
        else:
            lbfgs(loss_and_flat_grad, get_weights(u_model), Struct(), 
                  maxIter=newton_iter, learningRate=lr_lbfgs_sa)
            
#L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, 
                           x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, lmbda, eps):
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

def predict(X_star, u_x_model, u_model):
    with tf.device('/GPU:0'):
        u_star, _ = u_x_model(u_model, X_star[:,0:1], X_star[:,1:2])
    return u_star

sizes_w = []
sizes_b = []
with tf.device('/GPU:0'):
    for q, width in enumerate(layers_sa):
        if q != 1:
            sizes_w.append(int(width * layers_sa[1]))
            sizes_b.append(int(width if q != 0 else layers_sa[1]))

###############################################################################
total_time_1 = time.time()
for i, neuron in enumerate(neurons):
    print(print_seperator)
    ac_train = np.delete(ac_train, np.where(np.all(ac_train == neuron, 
                                                   axis=1))[0], axis=0)    
    ###########################################################################
    # SA-PINN to be used as activation function
    lmbda, eps = neuron
    
    col_weights = tf.Variable(    tf.random.uniform([N_f, 1], seed=seed))
    u_weights   = tf.Variable(100*tf.random.uniform([N0, 1],  seed=seed))

    u_model = neural_net(layers_sa)
    
    t1 = time.time()
    fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, 
    tf_iter=epochs_adam_sa, newton_iter=epochs_lbfgs_sa, lmbda=float(lmbda), 
    eps=float(eps))
    t2 = time.time()
    print(f"PINN time: {(t2-t1)/60} minutes\n")
    
    # Convert TensorFlow model to PyTorch model
    for j in range(len(SA_w)):
        SA_w[j] = torch.Tensor(u_model.layers[j].get_weights()[0].T)
        
    for k in range(len(SA_b)):
        SA_b[k] = torch.Tensor(u_model.layers[k].get_weights()[1])
    
    PINN = P(SA_w, SA_b, layers_sa).to(device)   

    ###########################################################################    
    # GPT-PINN Training / Offline Generation
    c_initial  = c_init[i][0]
    
    train_out, train_out_xx, train_out_t, train_out_IC, train_out_BC_ub, \
    train_out_BC_lb, train_out_BC_diff, fhat, train_len, train_out_BC_ub_x,\
    train_out_BC_lb_x, train_out_BC_diff_x = inputs(PINN, xt_resid, out_full, 
    out_xx_full, out_t_full, out_IC, out_BC_ub, out_BC_lb, IC_xt, BC_xt_ub, 
    BC_xt_lb, i, out_test, xt_test, f_hat, xt_size, out_BC_diff, out_BC_ub_x, 
    out_BC_lb_x, out_BC_diff_x)
         
    t1 = time.time()
    largest_loss, largest_case, trained_c = offline_generation(ac_train, c_initial, 
    train_len, IC_size, BC_size, IC_u, train_out, train_out_t, train_out_xx, 
    train_out_IC, train_out_BC_ub, train_out_BC_lb, train_out_BC_diff, 
    train_out_BC_ub_x, train_out_BC_lb_x, train_out_BC_diff_x, fhat, 
    epochs_gpt_train, lr_gpt)
    t2 = time.time()
    
    generation_time[i] = (t2-t1)/60
    print(f"Generation time: {(t2-t1)/60} minutes")  
    
    ###########################################################################
    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        neurons[i+1] = largest_case
        
    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"Parameter Case: {largest_case}\n")
    
    if (i == number_of_neurons-1):
        break

total_time = (time.time() - total_time_1) / 3600

print(print_seperator)
print(f"Total Training Time: {total_time} Hours\n")
print(f"Activation Function Parameters: \n{neurons}\n")
print(f"Loss list: {loss_list}\n")

###############################################################################
#### Test ####

ac_test = ac_train[np.random.choice(len(ac_train), test_cases, replace=False)]

print("SGPT-PINN Testing Started")
test_gpt_time, test_gpt_soln = gpt_test(ac_test, train_out, train_out_t, train_out_xx, 
train_out_IC, fhat, train_len, IC_size, BC_size, IC_u, c_initial, epochs_gpt_test, 
lr_gpt, out_test, train_out_BC_ub, train_out_BC_lb, train_out_BC_diff, 
train_out_BC_ub_x, train_out_BC_lb_x, train_out_BC_diff_x)

test_gpt_losses = gpt_test_loss(ac_test, train_out, train_out_t, train_out_xx, 
train_out_IC, fhat, train_len, IC_size, BC_size, IC_u, c_initial, epochs_gpt_test, 
lr_gpt, out_test, train_out_BC_ub, train_out_BC_lb, train_out_BC_diff, 
train_out_BC_ub_x, train_out_BC_lb_x, train_out_BC_diff_x)
print("SGPT-PINN Testing Ended\n")

np.savetxt(data_dir+"/generation_time.dat", generation_time)
np.savetxt(data_dir+"/loss_list.dat",       loss_list)
np.savetxt(data_dir+"/neurons.dat",         neurons)
np.savetxt(data_dir+"/total_time.dat",      np.array([total_time]))

np.savetxt(data_dir+"/xt_resid.dat",        xt_resid.detach().cpu().numpy())
np.savetxt(data_dir+"/ac_test.dat",         ac_test)
np.savetxt(data_dir+"/xt_test.dat",         xt_test.cpu().numpy())

np.savetxt(data_dir+"/test_gpt_losses.dat", test_gpt_losses)
np.savetxt(data_dir+"/test_gpt_soln.dat",   test_gpt_soln)
np.savetxt(data_dir+"/test_gpt_time.dat",   test_gpt_time+total_time)

params = {"device":device,
          "domain": {"xi":-1, "xf":1, "ti":0, "tf":1}, 
          "data sizes": {"N_f":N_f, "N_test":xt_size, "BC_pts":BC_size, "N0":IC_size},
          "layers_pinn":layers_sa,
          "lr_adam_sa":lr_adam_sa,
          "lr_lbfgs_sa":lr_lbfgs_sa,
          "epochs_adam_sa":epochs_adam_sa,
          "epochs_lbfgs_sa":epochs_lbfgs_sa,
          "parameter size":len(ac_train_all),
          "number_of_neurons":number_of_neurons,
          "lr_gpt":lr_gpt,
          "epochs gpt train":epochs_gpt_train,
          "test cases":test_cases,
          "epochs gpt test":epochs_gpt_test,
          "seed":seed}

np.save(data_dir+"/params.npy", params) 

print(f"End: {datetime.now()}\n")