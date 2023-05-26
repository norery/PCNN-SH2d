import scipy.interpolate as interpolate
from SALib.sample import sobol_sequence
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from torch.optim.lr_scheduler import StepLR
import os
import seaborn as sns

# CUDA support
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Your device is: {}".format(device))

# torch.manual_seed(66)
# np.random.seed(66)
torch.set_default_dtype(torch.float64)


# np.random.seed(1234)


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
    def __init__(self, ExistModel=0, model_Dir=""):

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
        self.x_pde = torch.tensor(X_pde[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_pde = torch.tensor(X_pde[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_pde = torch.tensor(X_pde[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.x_ini1 = torch.tensor(X_ini1[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini1 = torch.tensor(X_ini1[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini1 = torch.tensor(X_ini1[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.x_ini2 = torch.tensor(X_ini2[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini2 = torch.tensor(X_ini2[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini2 = torch.tensor(X_ini2[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.u_ini1 = torch.tensor(u_ini1, dtype=torch.float64, requires_grad=True).to(device)
        self.u_ini2 = torch.tensor(u_ini2, dtype=torch.float64, requires_grad=True).to(device)

        self.x_ini1_s2 = torch.tensor(X_ini1_s2[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini1_s2 = torch.tensor(X_ini1_s2[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini1_s2 = torch.tensor(X_ini1_s2[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.x_ini2_s2 = torch.tensor(X_ini2_s2[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini2_s2 = torch.tensor(X_ini2_s2[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini2_s2 = torch.tensor(X_ini2_s2[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.u_ini1_s2 = torch.tensor(u_ini1_s2, dtype=torch.float64, requires_grad=True).to(device)
        self.u_ini2_s2 = torch.tensor(u_ini2_s2, dtype=torch.float64, requires_grad=True).to(device)

        # optimizers
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            max_iter=5000,
            max_eval=5000,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )

        self.lr_adam = 5.e-3
        self.opt_adam = torch.optim.Adam(self.dnn.parameters(), lr=self.lr_adam)
        self.scheduler = StepLR(self.opt_adam, step_size=100, gamma=0.99)

        self.LBFGS_iter = 0
        self.adam_iter = 0

        self.loss_adam_log = []
        self.loss_pde_adam_log = []
        self.loss_ini_adam_log = []
        self.L2_error_log = []

    def net_u(self, x, z, t):
        u = self.dnn(torch.cat((x, z, t), dim=1))

        return u

    def calc_res(self, x, z, t):
        u = self.net_u(x, z, t)

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_z = torch.autograd.grad(
            u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_zz = torch.autograd.grad(
            u_z,
            z,
            grad_outputs=torch.ones_like(u_z),
            retain_graph=True,
            create_graph=True,
        )[0]

        P = u_xx + u_zz

        alpha_true = 1

        u_t = \
            torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_tt = \
            torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[0]

        res_pde = u_tt - alpha_true ** 2 * P  # Scalar Wave equation

        return res_pde

    def loss_func(self, IfIni):

        mse_loss = nn.MSELoss()

        if IfIni:
            u_ini1_pred = self.net_u(self.x_ini1, self.z_ini1, self.t_ini1)
            u_ini2_pred = self.net_u(self.x_ini2, self.z_ini2, self.t_ini2)
            res_ini1 = u_ini1_pred - self.u_ini1
            res_ini2 = u_ini2_pred - self.u_ini2
            loss_ini = mse_loss(res_ini1, torch.zeros_like(res_ini1).to(device)) + mse_loss(res_ini2, torch.zeros_like(
                res_ini2).to(device))
            loss = loss_ini
            loss_pde = torch.tensor(0.0).to(device)
        else:
            u_ini1_s2_pred = self.net_u(self.x_ini1_s2, self.z_ini1_s2, self.t_ini1_s2)
            u_ini2_s2_pred = self.net_u(self.x_ini2_s2, self.z_ini2_s2, self.t_ini2_s2)
            res_ini1 = u_ini1_s2_pred - self.u_ini1_s2
            res_ini2 = u_ini2_s2_pred - self.u_ini2_s2
            loss_ini = mse_loss(res_ini1, torch.zeros_like(res_ini1).to(device)) + mse_loss(res_ini2, torch.zeros_like(
                res_ini2).to(device))
            res_u_pred = self.calc_res(self.x_pde, self.z_pde, self.t_pde)
            res_pde = res_u_pred
            loss_pde = mse_loss(res_pde, torch.zeros_like(res_pde).to(device))
            loss = loss_pde + 5e3 * loss_ini

        return loss, loss_pde, loss_ini

    def train_adam(self, n_iters, IfIni=True, loop_iter=0):
        self.dnn.train()
        self.IfIni = IfIni
        self.loop_iter= loop_iter

        for epoch in range(n_iters):

            self.opt_adam.zero_grad()
            loss, loss_pde, loss_ini = self.loss_func(IfIni)
            loss.backward()
            self.opt_adam.step()

            self.scheduler.step()

            self.loss_adam_log.append(loss.detach().cpu().numpy())
            self.loss_pde_adam_log.append(loss_pde.detach().cpu().numpy())
            self.loss_ini_adam_log.append(loss_ini.detach().cpu().numpy())

            #####Defining a new training batch for both PDE and B.C input data
            self.adam_iter += 1
            if self.adam_iter % 1000 == 0:
                print("loop_iter %d, Adam iter %d, Loss: %.4e, loss_pde: %.4e, loss_ini: %.4e" % \
                      (self.loop_iter, self.adam_iter, loss.item(), loss_pde.item(), loss_ini.item()))

            if epoch % 2000 == 0:
                if IfIni:
                    fig_path = f'{fig_dir}/loop_{self.loop_iter}_ini_adam_{epoch}.png'
                    self.predict_eval(epoch, fig_path)
                    save_model_path = f'{save_checkpoints_dir}/loop_{self.loop_iter}_ini_adam_checkpoints_{epoch}.dump'
                    save_checkpoint(self.dnn, save_model_path)
                else:
                    fig_path = f'{fig_dir}/loop_{self.loop_iter}_adam_{epoch}.png'
                    self.predict_eval(epoch, fig_path)
                    save_model_path = f'{save_checkpoints_dir}/loop_{self.loop_iter}_adam_checkpoints_{epoch}.dump'
                    save_checkpoint(self.dnn, save_model_path)


    def closure(self):
        self.optimizer.zero_grad()
        loss_LBFGS, loss_pde_LBFGS, loss_ini_LBFGS = self.loss_func(self.IfIni)
        loss_LBFGS.backward()
        self.loss_LBFGS.append(loss_LBFGS.detach().cpu().numpy())
        self.loss_pde_LBFGS.append(loss_pde_LBFGS.detach().cpu().numpy())
        self.loss_ini_LBFGS.append(loss_ini_LBFGS.detach().cpu().numpy())


        self.LBFGS_iter += 1
        if self.LBFGS_iter % 1000 == 0:
            print("loop_iter %d, LBFGS_iter %d, Loss: %.4e, loss_pde: %.4e, loss_ini: %.4e" % \
                  (self.loop_iter, self.LBFGS_iter, loss_LBFGS.item(), loss_pde_LBFGS.item(), loss_ini_LBFGS.item()))

        if self.LBFGS_iter % 5000 == 0:
            if self.IfIni:
                fig_path = f'{fig_dir}/loop_{self.loop_iter}_ini_LBFGS_{self.LBFGS_iter}.png'
                self.predict_eval(self.LBFGS_iter, fig_path)
                save_model_path = f'{save_checkpoints_dir}/loop_{self.loop_iter}_ini_LBFGS_checkpoints_{self.LBFGS_iter}.dump'
                save_checkpoint(self.dnn, save_model_path)
            else:
                fig_path = f'{fig_dir}/loop_{self.loop_iter}_LBFGS_{self.LBFGS_iter}.png'
                self.predict_eval(self.LBFGS_iter, fig_path)
                save_model_path = f'{save_checkpoints_dir}/loop_{self.loop_iter}_LBFGS_checkpoints_{self.LBFGS_iter}.dump'
                save_checkpoint(self.dnn, save_model_path)

        return loss_LBFGS

    def train_LBFGS(self):
        self.loss_LBFGS = []
        self.loss_pde_LBFGS = []
        self.loss_ini_LBFGS = []
        self.L2_error_LBFGS_log = []

        self.dnn.train()
        self.optimizer.step(self.closure)

    def predict(self, X_evalt):
        x = torch.tensor(X_evalt[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        z = torch.tensor(X_evalt[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        t = torch.tensor(X_evalt[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.dnn.eval()
        u = self.dnn(torch.cat((x, z, t), dim=1))

        return u

    def predict_eval(self, epoch, figname):

        u_eval_01 = self.predict(X_eval01)
        u_eval_02 = self.predict(X_eval02)
        u_eval_03 = self.predict(X_eval03)
        u_eval_04 = self.predict(X_eval04)

        U_PINN_01 = u_eval_01.detach().cpu().numpy()
        U_PINN_02 = u_eval_02.detach().cpu().numpy()
        U_PINN_03 = u_eval_03.detach().cpu().numpy()
        U_PINN_04 = u_eval_04.detach().cpu().numpy()

        U_diff_01 = U_eval_all[0] - U_PINN_01
        U_diff_02 = U_eval_all[1] - U_PINN_02
        U_diff_03 = U_eval_all[2] - U_PINN_03
        U_diff_04 = U_eval_all[3] - U_PINN_04

        U_pinn_pred = [U_PINN_01, U_PINN_02, U_PINN_03, U_PINN_04]
        U_diff_all = [U_diff_01, U_diff_02, U_diff_03, U_diff_04]

        n_eval_time = 4
        shape = (3, n_eval_time)
        fig1 = plt.figure(figsize=(3 * shape[1], 3 * shape[0]))

        s = 10
        for it in range(4):
            plt.subplot2grid(shape, (0, it))
            plt.scatter(x_eval * xz_scl, z_eval * xz_scl, c=U_eval_all[it], alpha=0.9, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            plt.axis('equal')
            plt.colorbar()
            plt.title(f'epoch: {epoch}')

            plt.subplot2grid(shape, (1, it))
            plt.scatter(x_eval * xz_scl, z_eval * xz_scl, c=U_pinn_pred[it], alpha=0.9, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            plt.axis('equal')
            plt.colorbar()
            # plt.title(f'ini_PDE, epoch: {epoch}')

            plt.subplot2grid(shape, (2, it))
            plt.scatter(x_eval * xz_scl, z_eval * xz_scl, c=U_diff_all[it], alpha=0.9, edgecolors='none',
                        cmap='seismic', marker='o', s=s)
            plt.xticks([])
            plt.yticks([])
            plt.axis('equal')
            plt.colorbar()
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

    fig_dir = './fig_loop_5e3'
    save_checkpoints_dir = './save_loop_5e3'
    wavefields_path = 'wavefields'

    xz_scl = 600
    # PINN的x,z范围
    xmin_spec = 0
    xmax_spec = 1500 / xz_scl
    zmin_spec = 0
    zmax_spec = 1500 / xz_scl

    n_abs = 3
    nx = 100
    dx = xmax_spec / nx
    dz = zmax_spec / nx

    xmin = xmin_spec + dx * n_abs
    xmax = xmax_spec - dx * n_abs
    zmin = zmin_spec + dz * n_abs
    zmax = zmax_spec - dz * n_abs

    s_spec = 1e-4  # specfem time stepsize
    t01 = 4500 * s_spec  # initial disp. input at this time from spec
    t02 = 5000 * s_spec  # sec "initial" disp. input at this time from spec instead of enforcing initial velocity
    t03 = 9000 * s_spec  # test data for comparing specfem and trained PINNs
    t04 = 13000 * s_spec  # test data for comparing specfem and trained PINNs

    t_m = t04  # total time for PDE training.
    t_st = t01  # this is when we take the first I.C from specfem

    ###initial conditions for all events
    X_spec = np.loadtxt(wavefields_path + '/wavefield_grid_for_dumps.txt')

    X_spec = X_spec / xz_scl  # specfem works with meters unit so we need to convert them to Km
    X_spec[:, 0:1] = X_spec[:, 0:1]  # scaling the spatial domain
    X_spec[:, 1:2] = X_spec[:, 1:2]  # scaling the spatial domain
    xz_spec = np.concatenate((X_spec[:, 0:1], X_spec[:, 1:2]), axis=1)

    # uploading the wavefields from specfem
    wave_filed_dir_list = sorted(os.listdir(wavefields_path))
    U0 = [np.loadtxt(wavefields_path + '/' + f) for f in wave_filed_dir_list]

    """ First IC and Second IC """
    n_ini = 50
    xini_min = xmin
    xini_max = xmax
    x_ini = np.linspace(xini_min, xini_max, n_ini)
    z_ini = np.linspace(xini_min, xini_max, n_ini)
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
    u_ini1 = interpolate.griddata(xz_spec, U0[0], xz_ini, fill_value=0.0)  # [1600, 2]
    u_scl = max(abs(np.min(u_ini1)), abs(np.max(u_ini1)))
    u_ini1 = u_ini1.reshape(-1, 1) / u_scl
    u1_min = np.min(u_ini1)
    u1_max = np.max(u_ini1)
    u_color = max(abs(u1_min), abs(u1_max))
    print(f'shpae of U_ini1: {u_ini1.shape} === min: [{np.min(u_ini1)}] === max: [{np.max(u_ini1)}]')

    u_ini2 = interpolate.griddata(xz_spec, U0[1], xz_ini, fill_value=0.0)
    u_ini2 = u_ini2.reshape(-1, 1) / u_scl
    print(f'shpae of U_ini2: {u_ini2.shape} === min: [{np.min(u_ini2)}] === max: [{np.max(u_ini2)}]')

    """ First IC and Second IC """
    x_ini_s2 = np.linspace(xmin, xmax, n_ini)
    z_ini_s2 = np.linspace(zmin, zmax, n_ini)
    x_ini_s2_mesh, z_ini_s2_mesh = np.meshgrid(x_ini_s2, z_ini_s2)
    x_ini_s2 = x_ini_s2_mesh.reshape(-1, 1)
    z_ini_s2 = z_ini_s2_mesh.reshape(-1, 1)
    t_ini1_s2 = 0.0 * np.ones((n_ini ** 2, 1), dtype=np.float64)
    t_ini2_s2 = (t02 - t01) * np.ones((n_ini ** 2, 1), dtype=np.float64)
    # for enforcing the disp I.C
    X_ini1_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini1_s2), axis=1)  # [1600, 3]
    # for enforcing the sec I.C, another snapshot of specfem
    X_ini2_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini2_s2), axis=1)  # [1600, 3]
    xz_ini_s2 = np.concatenate((x_ini_s2, z_ini_s2), axis=1)  # [1600, 2]

    # uploading the wavefields from specfem
    u_ini1_s2 = interpolate.griddata(xz_spec, U0[0], xz_ini_s2, fill_value=0.0)  # [1600, 2]
    u_ini1_s2 = u_ini1_s2.reshape(-1, 1) / u_scl

    u_ini2_s2 = interpolate.griddata(xz_spec, U0[1], xz_ini_s2, fill_value=0.0)
    u_ini2_s2 = u_ini2_s2.reshape(-1, 1) / u_scl

    # wavefields for eval
    n_eval = 100
    x_eval = np.linspace(xmin, xmax, n_eval)
    z_eval = np.linspace(zmin, zmax, n_eval)
    x_eval_mesh, z_eval_mesh = np.meshgrid(x_eval, z_eval)
    x_eval = x_eval_mesh.reshape(-1, 1)
    z_eval = z_eval_mesh.reshape(-1, 1)
    xz_eval = np.concatenate((x_eval, z_eval), axis=1)  # [1600, 2]

    u_eval1_0 = interpolate.griddata(xz_spec, U0[0], xz_eval, fill_value=0.0)  # [1600, 2]
    u_eval1 = u_eval1_0.reshape(-1, 1) / u_scl
    u_eval2_0 = interpolate.griddata(xz_spec, U0[1], xz_eval, fill_value=0.0)
    u_eval2 = u_eval2_0.reshape(-1, 1) / u_scl
    u_eval3_0 = interpolate.griddata(xz_spec, U0[2], xz_eval, fill_value=0.0)  # Test data
    u_eval3 = u_eval3_0.reshape(-1, 1) / u_scl
    u_eval4_0 = interpolate.griddata(xz_spec, U0[3], xz_eval, fill_value=0.0)  # Test data
    u_eval4 = u_eval4_0.reshape(-1, 1) / u_scl

    X_eval01 = np.concatenate((x_eval, z_eval, 0 * np.ones_like(x_eval)), axis=1)
    X_eval02 = np.concatenate((x_eval, z_eval, (t02 - t01) * np.ones_like(x_eval)), axis=1)
    X_eval03 = np.concatenate((x_eval, z_eval, (t03 - t01) * np.ones_like(x_eval)), axis=1)
    X_eval04 = np.concatenate((x_eval, z_eval, (t04 - t01) * np.ones_like(x_eval)), axis=1)

    ################### plots of inputs for sum of the events
    ini_time = [0, round(t02 - t01, 4)]
    n_eval_time = len(ini_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(6 * shape[1], 5 * shape[0]))

    U_ini_plot = [u_ini1, u_ini2]

    for it in range(n_eval_time):
        plt.subplot2grid(shape, (0, it))
        plt.scatter(x_ini * xz_scl, z_ini * xz_scl, c=U_ini_plot[it], alpha=1, edgecolors='none',
                    cmap='seismic', marker='o', s=10, vmin=-1, vmax=1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar()
        plt.title('ini x t=' + str(ini_time[it]))

    # save_name2 = './figures/Specfem_wavefield_40x40.png'
    # plt.savefig(save_name2, dpi=300)
    plt.show()

    ################### plots of inputs for sum of the events
    eval_time = [0, round(t02 - t01, 4), round(t03 - t01, 4), round(t04 - t01, 4)]
    n_eval_time = len(eval_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(8 * shape[1], 5 * shape[0]))

    U_eval_all = [u_eval1, u_eval2, u_eval3, u_eval4]

    s = 10
    for it in range(len(eval_time)):
        plt.subplot2grid(shape, (0, it))
        plt.scatter(x_eval * xz_scl, z_eval * xz_scl, c=U_eval_all[it], alpha=1, edgecolors='none',
                    cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('equal')
        plt.colorbar()
        # plt.title('Specfem t=' + str(eval_time[it]))

    # save_name2 = './figures/Specfem_wavefield_40x40.png'
    # plt.savefig(save_name2, dpi=300)
    plt.show()

    ### PDE residuals
    batch_size = 10000
    n_pde = batch_size * 1
    print('kernel_size', ':', batch_size)
    X_pde_sobol = sobol_sequence.sample(n_pde + 1, 3)[1:, :]
    x_pde = X_pde_sobol[:, 0] * (xmax - xmin) + xmin
    z_pde = X_pde_sobol[:, 1] * (zmax - zmin) + zmin
    t_pde = X_pde_sobol[:, 2] * (t_m - t_st)
    X_pde = np.concatenate((x_pde.reshape(-1, 1), z_pde.reshape(-1, 1), t_pde.reshape(-1, 1)), axis=1)
    layers = [3] + [30] * 5 + [1]  # layers for the NN approximating the scalar acoustic potential

    #  # train
    print("====== Start train Now ... =======")
    # train it, if networks are not provided
    L2_min_adam_all = []
    L2_min_adam_ini_all = []
    L2_min_LBFGS_ini_all = []
    for i in range(1):
        checkpoints_path = f"{save_checkpoints_dir}/loop_{i}_ini_LBFGS_checkpoints_5000.dump"
        model = PhysicsInformedNN()
        model.train_adam(n_iters=10001, IfIni=True, loop_iter=i)
        print('============================================================')

        model.train_LBFGS()
        print('============================================================')

        model = PhysicsInformedNN(ExistModel=1, model_Dir=checkpoints_path)
        model.train_adam(n_iters=10001, IfIni=False, loop_iter=i)
        print('============================================================')

