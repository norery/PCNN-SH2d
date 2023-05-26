import scipy.interpolate as interpolate
import os
import seaborn as sns
from SALib.sample import sobol_sequence
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import random
import pickle

# CUDA support
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Your device is: {}".format(device))

torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float64)
np.random.seed(1234)


def initialize_weights(module):
    """starting from small initialized parameters"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


# the deep neural network
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))

        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


# the physics-guided neural network
class PhysicsInformedNN:
    def __init__(self, XX, layers, ExistModel=0, model_Dir=""):

        # Define layers
        self.layers = layers

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        if ExistModel == 0:
            # initialize weights if networks are not provided
            self.dnn.apply(initialize_weights)

        else:
            # load previous model
            self.dnn = load_checkpoint(self.dnn, model_Dir)

        # input data
        self.x = torch.tensor(XX[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z = torch.tensor(XX[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t = torch.tensor(XX[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        # optimizers
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )

        self.lr_adam = 5.e-3
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=self.lr_adam)
        self.scheduler = StepLR(self.optimizer_adam, step_size=1000, gamma=0.90)

        self.LBFGS_iter = 0
        self.adam_iter = 0

    def loss_func(self):
        u = self.dnn(torch.cat((self.x, self.z, self.t), dim=1))

        u_x = torch.autograd.grad(
            u, self.x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            self.x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_z = torch.autograd.grad(
            u, self.z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_zz = torch.autograd.grad(
            u_z,
            self.z,
            grad_outputs=torch.ones_like(u_z),
            retain_graph=True,
            create_graph=True,
        )[0]

        lapl = u_xx + u_zz
        P = u_z

        alpha_true = 1

        u_t = \
        torch.autograd.grad(u, self.t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_tt = \
        torch.autograd.grad(u_t, self.t, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[0]

        eq = u_tt - alpha_true ** 2 * lapl  # Scalar Wave equation

        mse_loss = nn.MSELoss()

        # Calc loss
        res_pde = eq[:N1, 0:1]
        res_ini1 = u[N1:(N1 + N2), 0:1] - u_ini1
        res_ini2 = u[(N1 + N2):(N1 + N2 + N3), 0:1] - u_ini2
        res_top = P[(N1 + N2 + N3):, 0:1]

        loss_pde = (mse_loss(res_pde, torch.zeros_like(res_pde).to(device)))
        loss_ini1 = (
                mse_loss(res_ini1, torch.zeros_like(res_ini1).to(device))
        )

        loss_ini2 = (
                mse_loss(res_ini2, torch.zeros_like(res_ini2).to(device))
        )
        loss_top = (mse_loss(res_top, torch.zeros_like(res_top).to(device)))

        loss = loss_pde + 1e5 * (loss_ini1 + loss_ini2) + 1e3 * loss_top

        return loss, loss_pde, loss_ini1, loss_ini2, loss_top


    def train_adam(self, n_iters):
        self.dnn.train()

        bbn = 0
        loss_adam = []
        loss_pde_adam = []
        loss_ini1_adam = []
        loss_ini2_adam = []
        loss_top_adam = []
        for epoch in range(n_iters):

            self.optimizer_adam.zero_grad()
            loss, loss_pde, loss_ini1, loss_ini2, loss_top = self.loss_func()
            loss.backward()
            self.optimizer_adam.step()

            # self.scheduler.step()

            loss_adam.append(loss.detach().cpu().numpy())
            loss_pde_adam.append(loss_pde.detach().cpu().numpy())
            loss_ini1_adam.append(loss_ini1.detach().cpu().numpy())
            loss_ini2_adam.append(loss_ini2.detach().cpu().numpy())
            loss_top_adam.append(loss_top.detach().cpu().numpy())

            #####Defining a new training batch for both PDE and B.C input data
            # if epoch % 1000 == 0:
            #     bbn = bbn + 1
            #     X_input = np.concatenate((X_pde[bbn * batch_size:(bbn + 1) * batch_size], X_ini1, X_ini2, X_top), axis=0)
            #
            #     # update the input data
            #     self.x = torch.tensor(X_input[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
            #     self.z = torch.tensor(X_input[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
            #     self.t = torch.tensor(X_input[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

            self.adam_iter += 1
            if self.adam_iter % 200 == 0:
                print("Iter %d, Loss: %.4e, loss_pde: %.4e, loss_ini1: %.4e, loss_ini2: %.4e, loss_top: %.4e" % \
                      (self.adam_iter, loss.item(), loss_pde.item(), loss_ini1.item(), loss_ini2.item(), loss_top.item()))

            if epoch % 1000 == 0:
                fig_path = f'{fig_dir}/adam_{epoch}.png'
                self.predict_eval(epoch, fig_path)
                save_model_path = f'{save_checkpoints_dir}/adam_checkpoints_{epoch}.dump'
                save_checkpoint(self.dnn, save_model_path)

        return loss_adam, loss_pde_adam, loss_ini1_adam, loss_ini2_adam, loss_top_adam

    def closure(self):
        self.optimizer.zero_grad()
        loss_LBFGS, loss_pde_LBFGS, loss_ini1_LBFGS, loss_ini2_LBFGS, loss_top_LBFGS = self.loss_func()
        loss_LBFGS.backward()
        self.loss_LBFGS.append(loss_LBFGS.detach().cpu().numpy())
        self.loss_pde_LBFGS.append(loss_pde_LBFGS.detach().cpu().numpy())
        self.loss_ini1_LBFGS.append(loss_ini1_LBFGS.detach().cpu().numpy())
        self.loss_ini2_LBFGS.append(loss_ini2_LBFGS.detach().cpu().numpy())
        self.loss_top_LBFGS.append(loss_top_LBFGS.detach().cpu().numpy())

        self.LBFGS_iter += 1
        if self.LBFGS_iter % 200 == 0:
            print("Iter %d, Loss: %.4e, loss_pde: %.4e, loss_ini1: %.4e, loss_ini2: %.4e, loss_top: %.4e" % \
                  (self.LBFGS_iter, loss_LBFGS.item(), loss_pde_LBFGS.item(), loss_ini1_LBFGS.item(), loss_ini2_LBFGS.item(), loss_top_LBFGS.item()))

        if self.LBFGS_iter % 1000 == 0:
            fig_path = f'{fig_dir}/LBFGS_{self.LBFGS_iter}.png'
            self.predict_eval(self.LBFGS_iter, fig_path)
            save_model_path = f'{save_checkpoints_dir}/LBFGS_checkpoints_{self.LBFGS_iter}.dump'
            save_checkpoint(self.dnn, save_model_path)
        return loss_LBFGS

    def train_LBFGS(self):
        self.loss_LBFGS = []
        self.loss_pde_LBFGS = []
        self.loss_ini1_LBFGS = []
        self.loss_ini2_LBFGS = []
        self.loss_top_LBFGS = []

        self.dnn.train()
        self.optimizer.step(self.closure)

        return self.loss_LBFGS, self.loss_pde_LBFGS, self.loss_ini1_LBFGS, self.loss_ini2_LBFGS, self.loss_top_LBFGS

    def predict(self, X_evalt):
        x = torch.tensor(X_evalt[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        z = torch.tensor(X_evalt[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        t = torch.tensor(X_evalt[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.dnn.eval()
        u = self.dnn(torch.cat((x, z, t), dim=1))

        return u

    def predict_eval(self, epoch, figname):

        X_eval01 = np.concatenate((x_eval, z_eval, np.zeros_like(x_eval)), axis=1)
        X_eval02 = np.concatenate((x_eval, z_eval, (t02 - t01) * np.ones_like(x_eval)), axis=1)
        X_eval03 = np.concatenate((x_eval, z_eval, (t03 - t01) * np.ones_like(x_eval)), axis=1)
        X_eval04 = np.concatenate((x_eval, z_eval, (t04 - t01) * np.ones_like(x_eval)), axis=1)

        u_eval_01 = self.predict(X_eval01)
        u_eval_02 = self.predict(X_eval02)
        u_eval_03 = self.predict(X_eval03)
        u_eval_04 = self.predict(X_eval04)

        U_PINN_01 = u_eval_01.detach().cpu().numpy()
        U_PINN_02 = u_eval_02.detach().cpu().numpy()
        U_PINN_03 = u_eval_03.detach().cpu().numpy()
        U_PINN_04 = u_eval_04.detach().cpu().numpy()

        U_diff_01 = U_specfem_all[0] - U_PINN_01
        U_diff_02 = U_specfem_all[1] - U_PINN_02
        U_diff_03 = U_specfem_all[2] - U_PINN_03
        U_diff_04 = U_specfem_all[3] - U_PINN_04

        U_pinn_pred = [U_PINN_01, U_PINN_02, U_PINN_03, U_PINN_04]
        U_diff_all = [U_diff_01, U_diff_02, U_diff_03, U_diff_04]

        # TODO: need to convert tensor to numpy to plot figures
        eval_time = [0, round(t02 - t01, 4), round(t03 - t01, 4), round(t04 - t01, 4)]
        n_eval_time = len(eval_time)
        shape = (3, n_eval_time)
        fig1 = plt.figure(figsize=(3 * shape[1], 3 * shape[0]))

        for it in range(len(eval_time)):
            plt.subplot2grid(shape, (0, it))
            plt.scatter(x_eval, z_eval, c=U_specfem_all[it], alpha=0.9, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.title(f'ini_PDE, epoch: {epoch}')

            plt.subplot2grid(shape, (1, it))
            plt.scatter(x_eval, z_eval, c=U_pinn_pred[it], alpha=0.9, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            # plt.title(f'ini_PDE, epoch: {epoch}')

            plt.subplot2grid(shape, (2, it))
            plt.scatter(x_eval, z_eval, c=U_diff_all[it], alpha=0.9, edgecolors='none',
                        cmap='seismic', marker='o', s=s)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            # plt.title(f'ini_PDE, epoch: {epoch}')
        plt.savefig(figname, dpi=100)
        # plt.show()
        plt.close()


def save_checkpoint(model, save_dir):
    """save model and optimizer"""
    torch.save({"model_state_dict": model.state_dict()}, save_dir)
    print("Pretrained model saved!")


def load_checkpoint(model, save_dir):
    """load model and optimizer"""
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Pretrained model loaded!")

    return model


if __name__ == '__main__':
    fig_dir = 'log/fig_z1200'
    save_checkpoints_dir = 'log/save_z1200'
    wavefields_path = 'wavefields'

    xz_scl = 3000
    # PINN的x,z范围
    xmin_spec = 0
    xmax_spec = 1500 / xz_scl
    zmin_spec = 0
    zmax_spec = 1200 / xz_scl

    n_abs = 3
    nx = 100
    dx = xmax_spec / nx
    dz = zmax_spec / nx

    xmin = xmin_spec + dx * n_abs
    xmax = xmax_spec - dx * n_abs
    zmin = zmin_spec + dz * n_abs
    zmax = zmax_spec

    s_spec = 5e-5  # specfem time stepsize
    t01 = 3000 * s_spec  # initial disp. input at this time from spec
    t02 = 3500 * s_spec  # sec "initial" disp. input at this time from spec instead of enforcing initial velocity
    t03 = 6000 * s_spec  # test data for comparing specfem and trained PINNs
    t04 = 9000 * s_spec  # test data for comparing specfem and trained PINNs

    t_m = t04  # total time for PDE training.
    t_st = t01  # this is when we take the first I.C from specfem

    ###  x and z coordination, [160801]
    xz = np.loadtxt(f'{wavefields_path}/wavefield_grid_for_dumps.txt')

    xz = xz / xz_scl  # specfem works with meters unit so we need to convert them to Km
    xz[:, 0:1] = xz[:, 0:1]
    xz[:, 1:2] = xz[:, 1:2]

    """ First IC and Second IC """
    n_ini = 50
    xini_min = xmin
    xini_max = xmax
    zini_min = zmin
    zini_max = zmax
    x_ini = np.linspace(xini_min, xini_max, n_ini)
    z_ini = np.linspace(zini_min, zini_max, n_ini)
    x_ini_mesh, z_ini_mesh = np.meshgrid(x_ini, z_ini)
    x_ini = x_ini_mesh.reshape(-1, 1)
    z_ini = z_ini_mesh.reshape(-1, 1)
    t_ini1 = 0.0 * np.ones((n_ini ** 2, 1), dtype=np.float64)
    t_ini2 = (t02 - t01) * np.ones((n_ini ** 2, 1), dtype=np.float64)
    # for enforcing the disp I.C
    X_ini1 = np.concatenate((x_ini, z_ini, t_ini1), axis=1)  # [1600, 3]
    # for enforcing the sec I.C, another snapshot of specfem
    X_ini2 = np.concatenate((x_ini, z_ini, t_ini2), axis=1)  # [1600, 3]
    xz_ini = np.concatenate((x_ini, z_ini), axis=1)  # [1600, 2]

    # uploading the wavefields from specfem
    wave_filed_dir_list = sorted(os.listdir(wavefields_path))
    U0 = [np.loadtxt(wavefields_path + '/' + f) for f in wave_filed_dir_list]

    u_ini1 = interpolate.griddata(xz, U0[0], xz_ini, fill_value=0.0)  # [1600, 2]
    u_scl = max(abs(np.min(u_ini1)), abs(np.max(u_ini1)))
    u_ini1 = u_ini1.reshape(-1, 1) / u_scl
    u1_min = np.min(u_ini1)
    u1_max = np.max(u_ini1)
    u_color = max(abs(u1_min), abs(u1_max))
    print(f'shpae of U_ini1: {u_ini1.shape} === min: [{np.min(u_ini1)}] === max: [{np.max(u_ini1)}]')

    u_ini2 = interpolate.griddata(xz, U0[1], xz_ini, fill_value=0.0)
    u_ini2 = u_ini2.reshape(-1, 1) / u_scl
    print(f'shpae of U_ini2: {u_ini2.shape} === min: [{np.min(u_ini2)}] === max: [{np.max(u_ini2)}]')

    # wavefields for eval
    n_eval = 100
    x_eval = np.linspace(xmin, xmax, n_eval)
    z_eval = np.linspace(zmin, zmax, n_eval)
    x_eval_mesh, z_eval_mesh = np.meshgrid(x_eval, z_eval)
    x_eval = x_eval_mesh.reshape(-1, 1)
    z_eval = z_eval_mesh.reshape(-1, 1)
    xz_eval = np.concatenate((x_eval, z_eval), axis=1)  # [1600, 2]

    u_eval1_0 = interpolate.griddata(xz, U0[0], xz_eval, fill_value=0.0)  # [1600, 2]
    u_eval1 = u_eval1_0.reshape(-1, 1) / u_scl
    u_eval2_0 = interpolate.griddata(xz, U0[1], xz_eval, fill_value=0.0)
    u_eval2 = u_eval2_0.reshape(-1, 1) / u_scl
    u_eval3_0 = interpolate.griddata(xz, U0[2], xz_eval, fill_value=0.0)  # Test data
    u_eval3 = u_eval3_0.reshape(-1, 1) / u_scl
    u_eval4_0 = interpolate.griddata(xz, U0[3], xz_eval, fill_value=0.0)  # Test data
    u_eval4 = u_eval4_0.reshape(-1, 1) / u_scl

    ################### plots of inputs for sum of the events
    eval_time = [0, round(t02 - t01, 4), round(t03 - t01, 4), round(t04 - t01, 4)]
    n_eval_time = len(eval_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(3 * shape[1], 3 * shape[0]))

    U_specfem_all = [u_eval1, u_eval2, u_eval3, u_eval4]

    s = 5
    for it in range(len(eval_time)):
        plt.subplot2grid(shape, (0, it))
        plt.scatter(x_eval * xz_scl, z_eval * xz_scl, c=U_specfem_all[it], alpha=1, edgecolors='none',
                    cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
        # plt.xticks([])
        # plt.yticks([])
        plt.colorbar()
        # plt.title('Specfem t=' + str(eval_time[it]))

    # save_name2 = './figures/Specfem_wavefield_40x40.png'
    # plt.savefig(save_name2, dpi=300)
    plt.show()
    ###############################################################

    u_ini1 = torch.tensor(u_ini1).to(device)
    u_ini2 = torch.tensor(u_ini2).to(device)
    u_eval1 = torch.tensor(u_eval1).to(device)
    u_eval2 = torch.tensor(u_eval2).to(device)
    u_eval3 = torch.tensor(u_eval3).to(device)
    u_eval4 = torch.tensor(u_eval4).to(device)

    layers = [3] + [30] * 5 + [1]  # layers for the NN approximating the scalar acoustic potential

    ####  BCs: Free stress on top and no BC for other sides (absorbing)
    nx_top = 100
    nt_top = 50
    x_top = (np.random.rand(nx_top, 1) * (xmax - xmin) + xmin)
    t_top = np.random.rand(nt_top, 1) * (t_m - t_st)
    x_top_mesh, t_top_mesh = np.meshgrid(x_top, t_top)
    x_top = x_top_mesh.reshape(-1, 1)
    z_top = zmax * np.ones((x_top_mesh.reshape((-1, 1)).shape[0], 1))
    t_top = t_top_mesh.reshape(-1, 1)

    X_top = np.concatenate((x_top, z_top, t_top), axis=1)

    ### PDE residuals
    batch_size = 10000
    n_pde = batch_size * 1
    print('batch_size', ':', batch_size)
    X_pde = sobol_sequence.sample(n_pde + 1, 3)[1:, :]
    X_pde[:, 0] = (X_pde[:, 0] * (xmax - xmin) + xmin)
    X_pde[:, 1] = (X_pde[:, 1] * (zmax - zmin) + zmin)
    X_pde[:, 2] = X_pde[:, 2] * (t_m - t_st)

    N1 = batch_size
    N2 = X_ini1.shape[0]
    N3 = X_ini2.shape[0]

    X_input = np.concatenate((X_pde[0:batch_size], X_ini1, X_ini2, X_top), axis=0)

    #  # train
    # checkpoints_path = "save_checkpoints_x1500z1000/adam_20000.dump"
    print("====== Start train Now ... =======")
    # train it, if networks are not provided
    model = PhysicsInformedNN(X_input, layers)
    loss_adam, loss_pde_adam, loss_ini1_adam, loss_ini2_adam, loss_top_adam = model.train_adam(n_iters=20001)

    # load network if networks are provided
    loss_LBFGS, loss_pde_LBFGS, loss_ini1_LBFGS, loss_ini2_LBFGS, loss_top_LBFGS = model.train_LBFGS()
