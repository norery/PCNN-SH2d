import scipy.interpolate as interpolate
from SALib.sample import sobol_sequence
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.optim.lr_scheduler import StepLR
from functorch import jacrev, vmap, make_functional, grad, vjp
import timeit
import seaborn as sns

# CUDA support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Your device is: {}".format(device))
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        self.x_ini1 = torch.tensor(X_ini1[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini1 = torch.tensor(X_ini1[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini1 = torch.tensor(X_ini1[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.x_ini2 = torch.tensor(X_ini2[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini2 = torch.tensor(X_ini2[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini2 = torch.tensor(X_ini2[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.u_ini1 = torch.tensor(u_ini1[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.u_ini2 = torch.tensor(u_ini2[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)

        # optimizers
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            max_iter=5001,
            max_eval=5001,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )

        self.lr_adam = 5.e-3
        self.opt_adam = torch.optim.Adam(
            self.dnn.parameters(),
            lr=self.lr_adam)
        self.scheduler = StepLR(self.opt_adam, step_size=100, gamma=0.99)

        self.LBFGS_iter = 0
        self.adam_iter = 0

        self.K_ini1_log = []
        self.K_ini2_log = []

        self.loss_adam = []
        self.loss_ini_adam = []

        self.lambda_ini1_log = []
        self.lambda_ini2_log = []

    def net_u(self, x, z, t):
        u = self.dnn(torch.cat((x, z, t), dim=1))

        return u

    def loss_func(self):

        mse_loss = nn.MSELoss()

        u_ini1_pred = self.net_u(self.x_ini1, self.z_ini1, self.t_ini1)
        u_ini2_pred = self.net_u(self.x_ini2, self.z_ini2, self.t_ini2)

        # Calc loss
        res_ini1 = u_ini1_pred - self.u_ini1
        res_ini2 = u_ini2_pred - self.u_ini2

        loss_ini1 = mse_loss(res_ini1, torch.zeros_like(res_ini1).to(device))
        loss_ini2 = mse_loss(res_ini2, torch.zeros_like(res_ini2).to(device))

        loss = loss_ini1 + loss_ini2

        return loss, loss_ini1

    def train_adam(self, n_iters):
        self.dnn.train()

        start = timeit.default_timer()
        bbn = 0
        for epoch in range(n_iters):

            self.opt_adam.zero_grad()
            loss, loss_ini = self.loss_func()
            loss.backward()
            self.opt_adam.step()

            # self.scheduler.step()

            self.loss_adam.append(loss.detach().cpu().numpy())
            self.loss_ini_adam.append(loss_ini.detach().cpu().numpy())

            self.adam_iter += 1
            if self.adam_iter % 500 == 0:
                stop = timeit.default_timer()
                print('Time: ', stop - start)
                print("Iter %d, Loss: %.4e, loss_ini: %.4e" % (self.adam_iter, loss.item(), loss_ini.item()))

            if epoch % 2000 == 0:
                fig_path = f'{fig_dir}/adam_{epoch}.png'
                self.predict_eval(epoch, fig_path)
                save_model_path = f'{save_checkpoints_dir}/adam_checkpoints_{epoch}.dump'
                save_checkpoint(self.dnn, save_model_path)

    def closure(self):

        self.optimizer.zero_grad()
        loss_LBFGS, loss_ini_LBFGS = self.loss_func()
        loss_LBFGS.backward()
        self.loss_LBFGS.append(loss_LBFGS.detach().cpu().numpy())
        self.loss_ini_LBFGS.append(loss_ini_LBFGS.detach().cpu().numpy())

        if self.LBFGS_iter % 500 == 0:
            print("Iter %d, Loss: %.4e, loss_ini: %.4e" % (self.LBFGS_iter, loss_LBFGS.item(), loss_ini_LBFGS.item()))

        if self.LBFGS_iter % 1000 == 0:
            fig_path = f'{fig_dir}/LBFGS_{self.LBFGS_iter}.png'
            self.predict_eval(self.LBFGS_iter, fig_path)
            save_model_path = f'{save_checkpoints_dir}/LBFGS_checkpoints_{self.LBFGS_iter}.dump'
            save_checkpoint(self.dnn, save_model_path)

        self.LBFGS_iter += 1
        return loss_LBFGS

    def train_LBFGS(self, calc_NTK=True, update_lambda=True):
        self.loss_LBFGS = []
        self.loss_ini_LBFGS = []

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

        x_eval01 = np.concatenate((x_eval, z_eval, 0 * np.ones_like(x_eval)), axis=1)
        x_eval02 = np.concatenate((x_eval, z_eval, (t02 - t01) * np.ones((x_eval.shape[0], 1))), axis=1)
        x_eval03 = np.concatenate((x_eval, z_eval, (t03 - t01) * np.ones((x_eval.shape[0], 1))), axis=1)
        x_eval04 = np.concatenate((x_eval, z_eval, (t04 - t01) * np.ones((x_eval.shape[0], 1))), axis=1)

        u_eval_01 = self.predict(x_eval01)
        u_eval_02 = self.predict(x_eval02)
        u_eval_03 = self.predict(x_eval03)
        u_eval_04 = self.predict(x_eval04)

        u_eval_01 = u_eval_01.detach().cpu().numpy()
        u_eval_02 = u_eval_02.detach().cpu().numpy()
        u_eval_03 = u_eval_03.detach().cpu().numpy()
        u_eval_04 = u_eval_04.detach().cpu().numpy()

        u_diff_01 = U_specfem_all[0] - u_eval_01
        u_diff_02 = U_specfem_all[1] - u_eval_02
        u_diff_03 = U_specfem_all[2] - u_eval_03
        u_diff_04 = U_specfem_all[3] - u_eval_04

        u_evalz_all = [u_eval_01, u_eval_02, u_eval_03, u_eval_04]
        u_diff_all = [u_diff_01, u_diff_02, u_diff_03, u_diff_04]

        # TODO: need to convert tensor to numpy to plot figures
        eval_time = [0, round(t02 - t01, 4), round(t03 - t01, 4), round(t04 - t01, 4)]
        n_eval_time = len(eval_time)
        shape = (3, n_eval_time)
        fig1 = plt.figure(figsize=(3 * shape[1], 3 * shape[0]))

        u_color = 1
        s = 1
        for it in range(len(eval_time)):
            plt.subplot2grid(shape, (0, it))
            plt.scatter(x_eval, z_eval, c=U_specfem_all[it], alpha=1, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
            plt.xticks([])
            plt.yticks([])
            plt.axis('equal')
            plt.colorbar()
            plt.title(f'epoch: {epoch}')

            plt.subplot2grid(shape, (1, it))
            plt.scatter(x_eval, z_eval, c=u_evalz_all[it], alpha=1, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.axis('equal')
            # plt.title('PINNs t=' + str(eval_time[it]))

            plt.subplot2grid(shape, (2, it))
            plt.scatter(x_eval, z_eval, c=u_diff_all[it], alpha=1, edgecolors='none',
                        cmap='seismic', marker='o', s=s)
            plt.xticks([])
            plt.yticks([])
            plt.axis('equal')
            plt.colorbar()

        plt.savefig(figname, dpi=100)
        plt.show()


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

    fig_dir = 'log/fig_ini'
    save_checkpoints_dir = 'log/save_ini'
    wavefields_path = 'wavefields'

    # PINN的x,z范围
    xz_scl = 3000
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
    t03 = 7000 * s_spec  # test data for comparing specfem and trained PINNs
    t04 = 9000 * s_spec  # test data for comparing specfem and trained PINNs

    t_st = t01
    t_m = t04

    ###initial conditions for all events
    X_spec = np.loadtxt(wavefields_path + '/wavefield_grid_for_dumps.txt')

    X_spec = X_spec / xz_scl  # specfem works with meters unit so we need to convert them to Km
    X_spec[:, 0:1] = X_spec[:, 0:1]  # scaling the spatial domain
    X_spec[:, 1:2] = X_spec[:, 1:2]  # scaling the spatial domain
    xz_spec = X_spec

    print(' ========================== start interp wavefields part ==========================')
    n_ini = 50
    x_ini = np.linspace(xmin, xmax, n_ini)
    z_ini = np.linspace(zmin, zmax, n_ini)
    x_ini_mesh, z_ini_mesh = np.meshgrid(x_ini, z_ini)
    x_ini = x_ini_mesh.reshape(-1, 1)
    z_ini = z_ini_mesh.reshape(-1, 1)
    xz_ini = np.concatenate((x_ini, z_ini), axis=1)  # [1600, 2]

    # uploading the wavefields from specfem
    wave_filed_dir_list = sorted(os.listdir(wavefields_path))
    U0 = [np.loadtxt(wavefields_path + '/' + f) for f in wave_filed_dir_list]

    u_ini1 = interpolate.griddata(xz_spec, U0[0], xz_ini, fill_value=0.0)  # [1600, 2]
    u_scl = max(abs(np.min(u_ini1)), abs(np.max(u_ini1)))
    u_ini1 = u_ini1.reshape(-1, 1) / u_scl
    print(f'shpae of U_ini1: {u_ini1.shape} === min: [{np.min(u_ini1)}] === max: [{np.max(u_ini1)}]')

    u_ini2 = interpolate.griddata(xz_spec, U0[1], xz_ini, fill_value=0.0)
    u_ini2 = u_ini2.reshape(-1, 1) / u_scl
    print(f'shpae of U_ini2: {u_ini2.shape} === min: [{np.min(u_ini2)}] === max: [{np.max(u_ini2)}]')

    t_ini1 = np.zeros_like(x_ini)
    t_ini2 = np.ones_like(x_ini) * (t02 - t01)
    X_ini1 = np.concatenate((x_ini, z_ini, t_ini1), axis=1)
    X_ini2 = np.concatenate((x_ini, z_ini, t_ini2), axis=1)

    ################### plots of inputs for sum of the events
    eval_time = [0, round(t02 - t01, 4)]
    n_eval_time = len(eval_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(2 * shape[1], 2 * shape[0]))

    U_ini_all = [u_ini1, u_ini2]

    s = 1
    for it in range(len(eval_time)):
        plt.subplot2grid(shape, (0, it))
        # plt.scatter(x_ini, z_ini, c=U_ini_all[it], alpha=0.9, edgecolors='none',
        #             cmap='coolwarm', marker='o', s=s, vmin=-1, vmax=1)
        plt.scatter(x_ini, z_ini, c=U_ini_all[it], cmap='coolwarm', s=s, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()
        plt.axis('equal')
        plt.title('Ini interp u, t=' + str(eval_time[it]))
    # save_name2 = './figures/Specfem_wavefield_100x100.png'
    # plt.savefig(save_name2, dpi=100)
    plt.show()

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

    ################### plots of inputs for sum of the events
    eval_time = [0, round(t02 - t01, 4), round(t03 - t01, 4), round(t04 - t01, 4)]
    n_eval_time = len(eval_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(3 * shape[1], 3 * shape[0]))

    U_specfem_all = [u_eval1, u_eval2, u_eval3, u_eval4]

    ###############################################################
    layers = [3] + [30] * 5 + [1]  # layers for the NN approximating the scalar acoustic potential

    #  # train
    print("====== Start train Now ... =======")
    # train it, if networks are not provided
    model = PhysicsInformedNN()
    model.train_adam(n_iters=10001)
    #     with open('save_checkpoints_x1500z1500/elastic_SH_Infinite_x1500z1500_10Hz_adam_loss.pickle', 'wb') as f:
    #         pickle.dump([loss_adam, loss_pde_adam, loss_ini_adam], f)

    loss_adam = model.loss_adam
    loss_ini_adam = model.loss_ini_adam

    fig1 = plt.figure(figsize=(6, 5))
    iters = np.arange(len(loss_adam))
    with sns.axes_style("darkgrid"):
        # plt.plot(iters, loss_adam, label='$\mathcal{L}_{Total}$')
        plt.plot(iters, loss_ini_adam, label='$\mathcal{L}_{ini}$')
        plt.yscale('log')
        plt.xlabel('adam iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/adam_loss.png')
        plt.show()

    # load network if networks are provided
    model.train_LBFGS()

    # with open('save_checkpoints_x1500z1500/elastic_SH_Infinite_x1500z1500_10Hz_LBFGS_loss.pickle', 'wb') as f:
    #         pickle.dump([loss_LBFGS, loss_pde_LBFGS, loss_ini_LBFGS], f)

    loss_LBFGS = model.loss_LBFGS
    loss_ini_LBFGS = model.loss_ini_LBFGS

    iters = np.arange(len(loss_LBFGS))
    with sns.axes_style("darkgrid"):
        # plt.plot(iters, loss_LBFGS, label='$\mathcal{L}_{Total}$')
        plt.plot(iters, loss_ini_LBFGS, label='$\mathcal{L}_{ini}$')
        plt.yscale('log')
        plt.xlabel('LBFGS iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/LBFGS_loss.png')
        plt.show()
