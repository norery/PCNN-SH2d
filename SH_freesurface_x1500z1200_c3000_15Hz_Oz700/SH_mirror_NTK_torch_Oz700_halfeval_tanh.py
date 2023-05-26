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
import torch.autograd.functional as F
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
        self.x_pde = torch.tensor(X_pde[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_pde = torch.tensor(X_pde[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_pde = torch.tensor(X_pde[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.x_ini1 = torch.tensor(X_ini1[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini1 = torch.tensor(X_ini1[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini1 = torch.tensor(X_ini1[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.x_ini2 = torch.tensor(X_ini2[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.z_ini2 = torch.tensor(X_ini2[:, 1:2], dtype=torch.float64, requires_grad=True).to(device)
        self.t_ini2 = torch.tensor(X_ini2[:, 2:3], dtype=torch.float64, requires_grad=True).to(device)

        self.u_ini1 = torch.tensor(u_ini1_p12[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)
        self.u_ini2 = torch.tensor(u_ini2_p12[:, 0:1], dtype=torch.float64, requires_grad=True).to(device)

        # optimizers
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            max_iter=50001,
            max_eval=50001,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )

        self.lr_adam = 5.e-3
        self.opt_adam = torch.optim.Adam(
            self.dnn.parameters(),
            lr=self.lr_adam)
        # self.scheduler = StepLR(self.opt_adam, step_size=100, gamma=0.99)

        self.LBFGS_iter = 0
        self.adam_iter = 0

        self.K_ini1_log = []
        self.K_ini2_log = []
        self.K_pde_log = []

        self.loss_adam = []
        self.loss_ini_adam = []
        self.loss_pde_adam = []
        self.L2_adam_error_log = []

        self.lambda_ini1_log = []
        self.lambda_ini2_log = []
        self.lambda_pde_log = []

    def net_u(self, x, z, t):
        u = self.dnn(torch.cat((x, z, t), dim=1))

        return u

    def calc_res_pde(self, x, z, t):

        u = self.net_u(x, z, t)

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
        )[0]
        u_z = torch.autograd.grad(
            u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_zz = torch.autograd.grad(
            u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True
        )[0]
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True
        )[0]

        res_pde = u_tt - (u_xx + u_zz)

        return res_pde

    def compute_ini_ntk(self, x, z, t):
        fnet, params = make_functional(self.dnn)

        def fnet_single(params, x, z, t):
            X = torch.cat((x, z, t), dim=1)
            u = fnet(params, X)

            return u

        jac1 = jacrev(fnet_single)(params, x, z, t)  # 只算第一个参数的？
        jac1_flat = [j.flatten(2) for j in jac1]

        jac2 = jacrev(fnet_single)(params, x, z, t)
        jac2_flat = [j.flatten(2) for j in jac2]

        # full
        # einsum_expr = 'Naf,Mbf->NMab'
        # trace
        einsum_expr = 'Naf,Maf->NM'
        # diagonal
        # einsum_expr = 'Naf,Maf->NMa'

        # Compute J(x1) @ J(x2).T
        # TODO compute full or trace or diagnal
        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1_flat, jac2_flat)])
        result = result.sum(0).reshape(kernel_size, kernel_size)

        return result

    def compute_pde_ntk(self, x, z, t):
        fnet, params = make_functional(self.dnn)
        index = torch.ones((kernel_size, 1)).to(device)

        def fnet_x(params, x, z, t):
            X = torch.cat((x, z, t), dim=1)
            u = fnet(params, X)

            return u

        def calc_vjp_x(params, x, z, t):
            (vjp0, vjpfunc) = vjp(fnet_x, params, x, z, t)

            (_, vjp_x, vjp_z, vjp_t) = vjpfunc(index)

            return vjp_x

        def calc_vjp_z(params, x, z, t):
            (vjp0, vjpfunc) = vjp(fnet_x, params, x, z, t)

            (_, vjp_x, vjp_z, vjp_t) = vjpfunc(index)

            return vjp_z

        def calc_vjp_t(params, x, z, t):
            (vjp0, vjpfunc) = vjp(fnet_x, params, x, z, t)

            (_, vjp_x, vjp_z, vjp_t) = vjpfunc(index)

            return vjp_t

        def calc_vjp_xx(params, x, z, t):
            (vjp0, vjpfunc) = vjp(calc_vjp_x, params, x, z, t)
            (_, vjp_xx, vjp_xz, vjp_xt) = vjpfunc(index)

            return vjp_xx

        def calc_vjp_zz(params, x, z, t):
            (vjp0, vjpfunc) = vjp(calc_vjp_z, params, x, z, t)
            (_, vjp_zx, vjp_zz, vjp_zt) = vjpfunc(index)

            return vjp_zz

        def calc_vjp_tt(params, x, z, t):
            (vjp0, vjpfunc) = vjp(calc_vjp_t, params, x, z, t)
            (_, vjp_tx, vjp_tz, vjp_tt) = vjpfunc(index)

            return vjp_tt

        def calc_vjp_res(params, x, z, t):
            vjp_xx = calc_vjp_xx(params, x, z, t)
            vjp_zz = calc_vjp_zz(params, x, z, t)
            vjp_tt = calc_vjp_tt(params, x, z, t)

            res = vjp_tt - (vjp_xx + vjp_zz)

            return res

        jac1 = jacrev(calc_vjp_res)(params, x, z, t)
        jac1_flat = [j.flatten(2) for j in jac1]

        jac2 = jacrev(calc_vjp_res)(params, x, z, t)
        jac2_flat = [j.flatten(2) for j in jac2]
        # jac2_flat = jac1_flat

        # full
        # einsum_expr = 'Naf,Mbf->NMab'
        # trace
        einsum_expr = 'Naf,Maf->NM'
        # diagonal
        # einsum_expr = 'Naf,Maf->NMa'

        # Compute J(x1) @ J(x2).T
        # TODO compute full or trace or diagnal
        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1_flat, jac2_flat)])
        result = result.sum(0).reshape(kernel_size, kernel_size)

        return result

    def loss_func(self):

        mse_loss = nn.MSELoss()

        res_u_pred = self.calc_res_pde(self.x_pde, self.z_pde, self.t_pde)
        u_ini1_pred = self.net_u(self.x_ini1, self.z_ini1, self.t_ini1)
        u_ini2_pred = self.net_u(self.x_ini2, self.z_ini2, self.t_ini2)

        # Calc loss
        res_ini1 = u_ini1_pred - self.u_ini1
        res_ini2 = u_ini2_pred - self.u_ini2

        loss_pde = mse_loss(res_u_pred, torch.zeros_like(res_u_pred).to(device))
        loss_ini1 = mse_loss(res_ini1, torch.zeros_like(res_ini1).to(device))
        loss_ini2 = mse_loss(res_ini2, torch.zeros_like(res_ini2).to(device))

        # loss = loss_ini1 + loss_ini2 + 1.e-5 * loss_pde
        loss = self.lambda_ini1 * loss_ini1 + self.lambda_ini2 * loss_ini2 + self.lambda_pde * loss_pde

        return loss, loss_ini1, loss_pde

    def train_adam(self, n_iters, calc_NTK=True, update_lambda=True):
        self.dnn.train()

        start = timeit.default_timer()
        bbn = 0
        for epoch in range(n_iters):

            if calc_NTK & (epoch % 1000 == 0):
                # print('calc NTK...')
                X_pde_NTK = sobol_sequence.sample(kernel_size + 1, 3)[1:, :]
                X_pde_NTK[:, 0] = X_pde_NTK[:, 0] * (xmax - xmin) + xmin
                X_pde_NTK[:, 1] = X_pde_NTK[:, 1] * (2 * zmax - zmin) + zmin
                X_pde_NTK[:, 2] = X_pde_NTK[:, 2] * (t_m - t_st)

                xz_ini_p12_ntk = sobol_sequence.sample(kernel_size + 1, 2)[1:, :]
                x_ini_p12_ntk = xz_ini_p12_ntk[:, 0:1] * (xmax - xmin) + xmin
                z_ini_p12_ntk = xz_ini_p12_ntk[:, 1:2] * (2 * zmax - zmin) + zmin
                x_ini_p12_ntk = x_ini_p12_ntk.reshape(-1, 1)
                z_ini_p12_ntk = z_ini_p12_ntk.reshape(-1, 1)
                t_ini1_ntk = np.zeros_like(x_ini_p12_ntk)
                t_ini2_ntk = np.ones_like(x_ini_p12_ntk) * (t02 - t01)
                X_ini1_NTK = np.concatenate((x_ini_p12_ntk, z_ini_p12_ntk, t_ini1_ntk), axis=1)
                X_ini2_NTK = np.concatenate((x_ini_p12_ntk, z_ini_p12_ntk, t_ini2_ntk), axis=1)

                self.x_pde_ntk = torch.tensor(X_pde_NTK[:, 0:1], dtype=torch.float64,
                                              requires_grad=True).to(device)
                self.z_pde_ntk = torch.tensor(X_pde_NTK[:, 1:2], dtype=torch.float64,
                                              requires_grad=True).to(device)
                self.t_pde_ntk = torch.tensor(X_pde_NTK[:, 2:3], dtype=torch.float64,
                                              requires_grad=True).to(device)
                self.x_ini1_ntk = torch.tensor(X_ini1_NTK[:, 0:1], dtype=torch.float64,
                                               requires_grad=True).to(device)
                self.z_ini1_ntk = torch.tensor(X_ini1_NTK[:, 1:2], dtype=torch.float64,
                                               requires_grad=True).to(device)
                self.t_ini1_ntk = torch.tensor(X_ini1_NTK[:, 2:3], dtype=torch.float64,
                                               requires_grad=True).to(device)

                self.x_ini2_ntk = torch.tensor(X_ini2_NTK[:, 0:1], dtype=torch.float64,
                                               requires_grad=True).to(device)
                self.z_ini2_ntk = torch.tensor(X_ini2_NTK[:, 1:2], dtype=torch.float64,
                                               requires_grad=True).to(device)
                self.t_ini2_ntk = torch.tensor(X_ini2_NTK[:, 2:3], dtype=torch.float64,
                                               requires_grad=True).to(device)

                self.K_ini1 = self.compute_ini_ntk(self.x_ini1_ntk, self.z_ini1_ntk, self.t_ini1_ntk)
                self.K_ini2 = self.compute_ini_ntk(self.x_ini2_ntk, self.z_ini2_ntk, self.t_ini2_ntk)
                self.K_pde = self.compute_pde_ntk(self.x_pde_ntk, self.z_pde_ntk, self.t_pde_ntk)

                self.K_ini1_log.append(self.K_ini1.detach().cpu().numpy())
                self.K_ini2_log.append(self.K_ini2.detach().cpu().numpy())
                self.K_pde_log.append(self.K_pde.detach().cpu().numpy())

                # print('calc NTK end!')

            if calc_NTK & update_lambda & (epoch % 1000 == 0):
                start_update_lambda = timeit.default_timer()
                # print('start update weights...')
                lambda_K_sum = torch.trace(self.K_ini1) + torch.trace(self.K_ini2) + torch.trace(self.K_pde)

                lambda_ini1 = lambda_K_sum / torch.trace(self.K_ini1)
                lambda_ini2 = lambda_K_sum / torch.trace(self.K_ini2)
                lambda_pde = lambda_K_sum / torch.trace(self.K_pde)

                # lambda_ini1 = torch.tensor(1e4, dtype=torch.float64, requires_grad=True).to(device)
                # lambda_ini2 = torch.tensor(1e4, dtype=torch.float64, requires_grad=True).to(device)
                # lambda_pde = torch.tensor(1, dtype=torch.float64, requires_grad=True).to(device)

                self.lambda_ini1 = torch.autograd.Variable(lambda_ini1, requires_grad=True)
                self.lambda_ini2 = torch.autograd.Variable(lambda_ini2, requires_grad=True)
                self.lambda_pde = torch.autograd.Variable(lambda_pde, requires_grad=True)

                self.lambda_ini1_log.append(self.lambda_ini1.detach().cpu().numpy())
                self.lambda_ini2_log.append(self.lambda_ini2.detach().cpu().numpy())
                self.lambda_pde_log.append(self.lambda_pde.detach().cpu().numpy())

                stop_update_lambda = timeit.default_timer()
                # print('Time: ', stop_update_lambda - start_update_lambda, 'end of update weights...')

            self.opt_adam.zero_grad()
            loss, loss_ini, loss_pde = self.loss_func()
            loss.backward()
            self.opt_adam.step()

            # self.scheduler.step()

            u_pred_t01 = model.predict(x_eval01)
            u_pred_t01 = u_pred_t01.detach().cpu().numpy()
            L2_error_t01 = np.linalg.norm(u_pred_t01 - u1_spec, 2) / np.linalg.norm(u1_spec, 2)
            u_pred_t02 = model.predict(x_eval02)
            u_pred_t02 = u_pred_t02.detach().cpu().numpy()
            L2_error_t02 = np.linalg.norm(u_pred_t02 - u2_spec, 2) / np.linalg.norm(u2_spec, 2)
            u_pred_t03 = model.predict(x_eval03)
            u_pred_t03 = u_pred_t03.detach().cpu().numpy()
            L2_error_t03 = np.linalg.norm(u_pred_t03 - u3_spec, 2) / np.linalg.norm(u3_spec, 2)
            u_pred_t04 = model.predict(x_eval04)
            u_pred_t04 = u_pred_t04.detach().cpu().numpy()
            L2_error_t04 = np.linalg.norm(u_pred_t04 - u4_spec, 2) / np.linalg.norm(u4_spec, 2)
            L2_error = (L2_error_t01 + L2_error_t02 + L2_error_t03 + L2_error_t04) / 4

            self.loss_adam.append(loss.detach().cpu().numpy())
            self.loss_ini_adam.append(loss_ini.detach().cpu().numpy())
            self.loss_pde_adam.append(loss_pde.detach().cpu().numpy())
            self.L2_adam_error_log.append(L2_error)

            self.adam_iter += 1
            if self.adam_iter % 500 == 0:
                stop = timeit.default_timer()
                print('Time: ', stop - start)
                print("Iter %d, Loss: %.4e, loss_ini: %.4e, loss_pde: %.4e" % \
                      (self.adam_iter, loss.item(), loss_ini.item(), loss_pde.item()))
                print(f'lambda_ini1: {self.lambda_ini1}, lambda_pde: {self.lambda_pde}')

            if epoch % 2000 == 0:
                fig_path = f'{fig_dir}/adam_{epoch}.png'
                self.predict_eval(epoch, fig_path)
                save_model_path = f'{save_dir}/adam_checkpoints_{epoch}.dump'
                save_checkpoint(self.dnn, save_model_path)

    def closure(self):

        self.optimizer.zero_grad()
        loss_LBFGS, loss_ini_LBFGS, loss_pde_LBFGS = self.loss_func()
        loss_LBFGS.backward()
        self.loss_LBFGS.append(loss_LBFGS.detach().cpu().numpy())
        self.loss_ini_LBFGS.append(loss_ini_LBFGS.detach().cpu().numpy())
        self.loss_pde_LBFGS.append(loss_pde_LBFGS.detach().cpu().numpy())

        u_pred_t01 = model.predict(x_eval01)
        u_pred_t01 = u_pred_t01.detach().cpu().numpy()
        L2_error_t01 = np.linalg.norm(u_pred_t01 - u1_spec, 2) / np.linalg.norm(u1_spec, 2)
        u_pred_t02 = model.predict(x_eval02)
        u_pred_t02 = u_pred_t02.detach().cpu().numpy()
        L2_error_t02 = np.linalg.norm(u_pred_t02 - u2_spec, 2) / np.linalg.norm(u2_spec, 2)
        u_pred_t03 = model.predict(x_eval03)
        u_pred_t03 = u_pred_t03.detach().cpu().numpy()
        L2_error_t03 = np.linalg.norm(u_pred_t03 - u3_spec, 2) / np.linalg.norm(u3_spec, 2)
        u_pred_t04 = model.predict(x_eval04)
        u_pred_t04 = u_pred_t04.detach().cpu().numpy()
        L2_error_t04 = np.linalg.norm(u_pred_t04 - u4_spec, 2) / np.linalg.norm(u4_spec, 2)
        L2_error = (L2_error_t01 + L2_error_t02 + L2_error_t03 + L2_error_t04) / 4

        self.L2_error_LBFGS_log.append(L2_error)

        if self.LBFGS_iter % 500 == 0:
            print("Iter %d, Loss: %.4e, loss_ini: %.4e, loss_pde: %.4e" % \
                  (self.LBFGS_iter, loss_LBFGS.item(), loss_ini_LBFGS.item(), loss_pde_LBFGS.item()))

            print(f'lambda_ini1: {self.lambda_ini1}, lambda_pde: {self.lambda_pde}')

        if self.LBFGS_iter % 1000 == 0:
            fig_path = f'{fig_dir}/LBFGS_{self.LBFGS_iter}.png'
            self.predict_eval(self.LBFGS_iter, fig_path)
            save_model_path = f'{save_dir}/LBFGS_checkpoints_{self.LBFGS_iter}.dump'
            save_checkpoint(self.dnn, save_model_path)

        self.LBFGS_iter += 1
        return loss_LBFGS

    def train_LBFGS(self, calc_NTK=True, update_lambda=True):
        self.loss_LBFGS = []
        self.loss_ini_LBFGS = []
        self.loss_pde_LBFGS = []
        self.L2_error_LBFGS_log = []

        self.calc_NTK = calc_NTK
        self.update_lambda = update_lambda
        self.bbn = 0

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

        u_eval_01 = self.predict(x_eval01)
        u_eval_02 = self.predict(x_eval02)
        u_eval_03 = self.predict(x_eval03)
        u_eval_04 = self.predict(x_eval04)

        u_eval_01 = u_eval_01.detach().cpu().numpy()
        u_eval_02 = u_eval_02.detach().cpu().numpy()
        u_eval_03 = u_eval_03.detach().cpu().numpy()
        u_eval_04 = u_eval_04.detach().cpu().numpy()

        u_diff_01 = U_eval_all[0] - u_eval_01
        u_diff_02 = U_eval_all[1] - u_eval_02
        u_diff_03 = U_eval_all[2] - u_eval_03
        u_diff_04 = U_eval_all[3] - u_eval_04

        u_evalz_all = [u_eval_01, u_eval_02, u_eval_03, u_eval_04]
        u_diff_all = [u_diff_01, u_diff_02, u_diff_03, u_diff_04]

        # TODO: need to convert tensor to numpy to plot figures
        eval_time = [0, round(t02 - t01, 4), round(t03 - t01, 4), round(t04 - t01, 4)]
        n_eval_time = len(eval_time)
        shape = (3, n_eval_time)
        fig1 = plt.figure(figsize=(2 * shape[1], 2 * shape[0]))

        u_color = 1
        s = 1
        for it in range(len(eval_time)):
            plt.subplot2grid(shape, (0, it))
            plt.scatter(x_eval, z_eval, c=U_eval_all[it], alpha=1, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
            plt.xticks([])
            plt.yticks([])
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.colorbar()
            plt.title(f'NTK epoch: {epoch}')

            plt.subplot2grid(shape, (1, it))
            plt.scatter(x_eval, z_eval, c=u_evalz_all[it], alpha=1, edgecolors='none',
                        cmap='seismic', marker='o', s=s, vmin=-u_color, vmax=u_color)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            # plt.title('PINNs t=' + str(eval_time[it]))

            plt.subplot2grid(shape, (2, it))
            plt.scatter(x_eval, z_eval, c=u_diff_all[it], alpha=1, edgecolors='none',
                        cmap='seismic', marker='o', s=s)
            plt.xticks([])
            plt.yticks([])
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
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

    fig_dir = 'log/fig_mirror_NTK'
    save_dir = 'log/save_mirror_NTK'
    wavefields_path = 'wavefields'
    kernel_size = 200

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
    t03 = 6000 * s_spec  # test data for comparing specfem and trained PINNs
    t04 = 9000 * s_spec  # test data for comparing specfem and trained PINNs

    t_st = t01
    t_m = t04

    ###initial conditions for all events
    X_spec = np.loadtxt(wavefields_path + '/wavefield_grid_for_dumps.txt')

    X_spec = X_spec / xz_scl  # specfem works with meters unit so we need to convert them to Km
    X_spec[:, 0:1] = X_spec[:, 0:1]  # scaling the spatial domain
    X_spec[:, 1:2] = X_spec[:, 1:2]  # scaling the spatial domain
    pos = np.where((X_spec[:, 0] > xmin) & (X_spec[:, 0] < xmax) & (X_spec[:, 1] > zmin))
    X_spec = X_spec[pos]
    x_spec = X_spec[:, 0:1]
    z_spec = X_spec[:, 1:2]
    xz_spec = np.concatenate((x_spec, z_spec), axis=1)
    x_eval = X_spec[:, 0:1]
    z_eval = X_spec[:, 1:2]

    print(' ========================== start original wavefields part ==========================')
    wave_filed_dir_list = sorted(os.listdir(wavefields_path))
    U0 = [np.loadtxt(wavefields_path + '/' + f) for f in wave_filed_dir_list]

    u1_spec = U0[0][pos]
    u_scl = max(abs(np.min(u1_spec)), abs(np.max(u1_spec)))
    u1_spec = u1_spec.reshape(-1, 1) / u_scl

    u2_spec = U0[1][pos].reshape(-1, 1) / u_scl
    u3_spec = U0[2][pos].reshape(-1, 1) / u_scl
    u4_spec = U0[3][pos].reshape(-1, 1) / u_scl

    x_eval01 = np.concatenate((x_eval, z_eval, 0 * np.ones_like(x_eval)), axis=1)
    x_eval02 = np.concatenate((x_eval, z_eval, (t02 - t01) * np.ones((x_eval.shape[0], 1))), axis=1)
    x_eval03 = np.concatenate((x_eval, z_eval, (t03 - t01) * np.ones((x_eval.shape[0], 1))), axis=1)
    x_eval04 = np.concatenate((x_eval, z_eval, (t04 - t01) * np.ones((x_eval.shape[0], 1))), axis=1)

#     x_spec_flipud = np.flipud(x_spec)
#     # pay attention
#     z_spec_p2 = z_spec + (zmax - zmin)

#     x_eval = np.concatenate((x_spec, x_spec_flipud), axis=0)
#     z_eval = np.concatenate((z_spec, z_spec_p2), axis=0)

#     u1_eval = np.concatenate((u1_spec, np.flipud(u1_spec)), axis=0)
#     u2_eval = np.concatenate((u2_spec, np.flipud(u2_spec)), axis=0)
#     u3_eval = np.concatenate((u3_spec, np.flipud(u3_spec)), axis=0)
#     u4_eval = np.concatenate((u4_spec, np.flipud(u4_spec)), axis=0)

    # ################### plots of inputs for sum of the events
#     eval_time = [0, round(t02 - t01, 4), round(t03 - t01, 4), round(t04 - t01, 4)]
#     n_eval_time = len(eval_time)
#     shape = (3, n_eval_time)

#     plt.figure(figsize=(3 * shape[1], 2 * shape[0]))

    U_eval_all = [u1_spec, u2_spec, u3_spec, u4_spec]
#     U_flipud_all = [np.flipud(u1_spec), np.flipud(u2_spec), np.flipud(u3_spec), np.flipud(u4_spec)]
#     U_eval_all = [u1_eval, u2_eval, u3_eval, u4_eval]

#     s = 1
#     for it in range(len(eval_time)):
#         plt.subplot2grid(shape, (0, it))
#         # plt.scatter(x_spec_flipud, z_spec, c=U_flipud_all[it], cmap='coolwarm', s=s, vmin=-1, vmax=1)
#         plt.scatter(x_spec_flipud * xz_scl, z_spec_p2 * xz_scl, c=U_flipud_all[it], cmap='coolwarm', edgecolors='none', marker='o',
#                     s=s, vmin=-1, vmax=1)
#         # plt.xticks([])
#         # plt.yticks([])
#         # plt.colorbar()
#         plt.axis('equal')
#         # plt.title('Specfem u, t=' + str(eval_time[it]))

#         plt.subplot2grid(shape, (1, it))
#         # plt.scatter(x_spec, z_spec, c=U_spec_all[it], cmap='coolwarm', s=s, vmin=-1, vmax=1)
#         plt.scatter(x_spec * xz_scl, z_spec * xz_scl, c=U_spec_all[it], cmap='coolwarm', edgecolors='none', marker='o',
#                     s=s, vmin=-1, vmax=1)
#         # plt.xticks([])
#         # plt.yticks([])
#         # plt.colorbar()
#         plt.axis('equal')
#         # plt.title('Specfem u, t=' + str(eval_time[it]))

#         plt.subplot2grid(shape, (2, it))
#         # plt.scatter(x_eval, z_eval, c=U_eval_all[it], cmap='coolwarm', s=s, vmin=-1, vmax=1)
#         plt.scatter(x_eval * xz_scl, z_eval * xz_scl, c=U_eval_all[it], cmap='coolwarm', edgecolors='none', marker='o',
#                     s=s, vmin=-1, vmax=1)
#         # plt.xticks([])
#         # plt.yticks([])
#         # plt.colorbar()
#         plt.axis('equal')
#         # plt.title('Specfem u, t=' + str(eval_time[it]))

#     # save_name2 = './figures/Specfem_wavefield.png'
#     # plt.savefig(save_name2, dpi=100)
#     plt.show()

    print(' ========================== start interp wavefields part ==========================')
    n_ini = 50
    x_ini = np.linspace(xmin, xmax, n_ini)
    z_ini = np.linspace(zmin, zmax, n_ini)
    add_zini = np.linspace(2 * (zmax - zmin), 0, n_ini)
    x_ini_mesh, z_ini_mesh = np.meshgrid(x_ini, z_ini)
    _, add_ini_mesh = np.meshgrid(x_ini, add_zini)
    add_zini = add_ini_mesh.reshape(-1, 1)
    x_ini = x_ini_mesh.reshape(-1, 1)
    z_ini = z_ini_mesh.reshape(-1, 1)
    xz_ini = np.concatenate((x_ini, z_ini), axis=1)  # [1600, 2]
    z_ini_p2 = z_ini + add_zini
    xz_ini_p2 = np.concatenate((x_ini, z_ini_p2), axis=1)

    # uploading the wavefields from specfem
    wave_filed_dir_list = sorted(os.listdir(wavefields_path))
    U0 = [np.loadtxt(wavefields_path + '/' + f) for f in wave_filed_dir_list]

    u_ini1 = interpolate.griddata(xz_spec, U0[0][pos], xz_ini, fill_value=0.0)  # [1600, 2]
    u_ini1 = u_ini1.reshape(-1, 1) / u_scl
    print(f'shpae of U_ini1: {u_ini1.shape} === min: [{np.min(u_ini1)}] === max: [{np.max(u_ini1)}]')

    u_ini2 = interpolate.griddata(xz_spec, U0[1][pos], xz_ini, fill_value=0.0)
    u_ini2 = u_ini2.reshape(-1, 1) / u_scl
    print(f'shpae of U_ini2: {u_ini2.shape} === min: [{np.min(u_ini2)}] === max: [{np.max(u_ini2)}]')

    u_ini1_p12 = np.concatenate((u_ini1, u_ini1), axis=0)
    u_ini2_p12 = np.concatenate((u_ini2, u_ini2), axis=0)
    U_ini = np.concatenate((u_ini1_p12, u_ini2_p12), axis=1)
    xz_ini_p12 = np.concatenate((xz_ini, xz_ini_p2))
    x_ini_p12 = xz_ini_p12[:, 0:1]
    z_ini_p12 = xz_ini_p12[:, 1:2]
    t_ini1_p12 = np.zeros_like(x_ini_p12)
    t_ini2_p12 = np.ones_like(x_ini_p12) * (t02 - t01)
    X_ini1 = np.concatenate((x_ini_p12, z_ini_p12, t_ini1_p12), axis=1)
    X_ini2 = np.concatenate((x_ini_p12, z_ini_p12, t_ini2_p12), axis=1)

    ################### plots of inputs for sum of the events
    eval_time = [0, round(t02 - t01, 4)]
    n_eval_time = len(eval_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(3 * shape[1], 2 * shape[0]))

    U_ini_all = [u_ini1_p12, u_ini2_p12]

    s = 1
    for it in range(len(eval_time)):
        plt.subplot2grid(shape, (0, it))
        # plt.scatter(x_ini, z_ini, c=U_ini_all[it], alpha=0.9, edgecolors='none',
        #             cmap='coolwarm', marker='o', s=s, vmin=-1, vmax=1)
        plt.scatter(x_ini_p12, z_ini_p12, c=U_ini_all[it], cmap='coolwarm', s=s, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()
        plt.axis('equal')
        plt.title('Ini interp u, t=' + str(eval_time[it]))
    # save_name2 = './figures/Specfem_wavefield_100x100.png'
    # plt.savefig(save_name2, dpi=100)
    plt.show()

    ###############################################################
    layers = [3] + [30] * 5 + [1]  # layers for the NN approximating the scalar acoustic potential

    batch_size = 20000
    n_pde = batch_size * 1
    print('batch_size', ':', batch_size)
    X_pde = sobol_sequence.sample(n_pde + 1, 3)[1:, :]
    X_pde[:, 0] = X_pde[:, 0] * xmax
    X_pde[:, 1] = X_pde[:, 1] * zmax * 2
    X_pde[:, 2] = X_pde[:, 2] * (t_m - t_st)

    #  # train
    checkpoints_path = "log/save_ini/LBFGS_checkpoints_5000.dump"
    print("====== Start train Now ... =======")
    # train it, if networks are not provided
    model = PhysicsInformedNN(ExistModel=1, model_Dir=checkpoints_path)
    model.train_adam(n_iters=20001, calc_NTK=True, update_lambda=True)

    loss_adam = model.loss_adam
    loss_ini_adam = model.loss_ini_adam
    loss_pde_adam = model.loss_pde_adam
    L2_loss_adam = model.L2_adam_error_log

    with open(f'{save_dir}/SH_adam_loss.pickle', 'wb') as f:
        pickle.dump([loss_adam, loss_ini_adam, loss_pde_adam, L2_loss_adam], f)

    fig1 = plt.figure(figsize=(6, 5))
    iters = np.arange(len(loss_adam))
    with sns.axes_style("darkgrid"):
        # plt.plot(iters, loss_adam, label='$\mathcal{L}_{Total}$')
        plt.plot(iters, loss_pde_adam, label='$\mathcal{L}_{pde}$')
        plt.plot(iters, loss_ini_adam, label='$\mathcal{L}_{ini}$')
        plt.plot(iters, L2_loss_adam, label='$\mathcal{L}_{2}$')
        plt.yscale('log')
        plt.xlabel('adam iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/adam_loss.png')
        plt.show()

    # NTK
    # Create loggers for eigenvalues of NTK
    lambda_K_ini1_log = []
    lambda_K_ini2_log = []
    lambda_K_pde_log = []

    # Restore the NTK
    K_ini1_list = model.K_ini1_log
    K_ini2_list = model.K_ini2_log
    K_pde_list = model.K_pde_log

    with open(f'{save_dir}/K_list_log.pickle', 'wb') as f:
        pickle.dump([K_ini1_list, K_ini2_list, K_pde_list], f)

    for k in range(len(K_ini1_list)):
        K_ini = K_ini1_list[k]
        K_ini2 = K_ini2_list[k]
        K_pde = K_pde_list[k]

        # Compute eigenvalues
        lambda_K_ini1, _ = np.linalg.eig(K_ini)
        lambda_K_ini2, _ = np.linalg.eig(K_ini2)
        lambda_K_pde, _ = np.linalg.eig(K_pde)
        # Sort in descresing order
        lambda_K_ini1 = np.sort(np.real(lambda_K_ini1))[::-1]
        lambda_K_ini2 = np.sort(np.real(lambda_K_ini2))[::-1]
        lambda_K_pde = np.sort(np.real(lambda_K_pde))[::-1]

        # Store eigenvalues
        lambda_K_ini1_log.append(lambda_K_ini1)
        lambda_K_ini2_log.append(lambda_K_ini2)
        lambda_K_pde_log.append(lambda_K_pde)

    #     Eigenvalues of NTK
    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.plot(lambda_K_ini1_log[0], label='$n=0$')
    plt.plot(lambda_K_ini1_log[1], '--', label='$n=1,000$')
    plt.plot(lambda_K_ini1_log[2], '--', label='$n=2,000$')
    plt.plot(lambda_K_ini1_log[3], '--', label='$n=3,00$')
    plt.plot(lambda_K_ini1_log[-2], '--', label='$n=9,00$')
    plt.plot(lambda_K_ini1_log[-1], '--', label='$n=10,000$')
    plt.xlabel('index')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(r'Eigenvalues of ${K}_{ini1}$')

    ax = plt.subplot(1, 3, 2)
    plt.plot(lambda_K_ini2_log[0], label='$n=0$')
    plt.plot(lambda_K_ini2_log[1], '--', label='$n=1,000$')
    plt.plot(lambda_K_ini2_log[2], '--', label='$n=2,000$')
    plt.plot(lambda_K_ini2_log[3], '--', label='$n=3,000$')
    plt.plot(lambda_K_ini2_log[-2], '--', label='$n=9,00$')
    plt.plot(lambda_K_ini2_log[-1], '--', label='$n=10,000$')
    plt.xlabel('index')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(r'Eigenvalues of ${K}_{ini2}$')

    plt.subplot(1, 3, 3)
    plt.plot(lambda_K_pde_log[0], label='$n=0$')
    plt.plot(lambda_K_pde_log[1], '--', label='$n=1,000$')
    plt.plot(lambda_K_pde_log[2], '--', label='$n=2,000$')
    plt.plot(lambda_K_pde_log[3], '--', label='$n=3,000$')
    plt.plot(lambda_K_pde_log[-2], '--', label='$n=9,00$')
    plt.plot(lambda_K_pde_log[-1], '--', label='$n=10,000$')
    plt.xlabel('index')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(r'Eigenvalues of ${K}_{pde}$')
    plt.savefig(f'{fig_dir}/Eigenvalues.png')
    plt.show()

    # Evolution of weights during training
    lambda_ini_log = model.lambda_ini1_log
    lambda_ini2_log = model.lambda_ini2_log
    lambda_pde_log = model.lambda_pde_log

    plt.figure(figsize=(6, 5))
    plt.plot(lambda_ini_log, label='$\lambda_{ini1}$')
    plt.plot(lambda_ini2_log, label='$\lambda_{ini2}$')
    plt.plot(lambda_pde_log, label='$\lambda_{pde}$')
    plt.xlabel('iterations')
    plt.ylabel('$\lambda$')
    plt.yscale('log')
    plt.legend()
    plt.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/lambda_log.png')
    plt.show()

    # load network if networks are provided
    model.train_LBFGS()

    loss_LBFGS = model.loss_LBFGS
    loss_ini_LBFGS = model.loss_ini_LBFGS
    loss_pde_LBFGS = model.loss_pde_LBFGS
    L2_loss_LBFGS = model.L2_error_LBFGS_log

    with open(f'{save_dir}/SH_LBFGS_loss.pickle', 'wb') as f:
        pickle.dump([loss_LBFGS, loss_pde_LBFGS, loss_ini_LBFGS, L2_loss_LBFGS], f)

    iters = np.arange(len(loss_LBFGS))
    with sns.axes_style("darkgrid"):
        # plt.plot(iters, loss_LBFGS, label='$\mathcal{L}_{Total}$')
        plt.plot(iters, loss_ini_LBFGS, label='$\mathcal{L}_{ini}$')
        plt.plot(iters, loss_pde_LBFGS, label='$\mathcal{L}_{pde}$')
        plt.plot(iters, L2_loss_LBFGS, label='$\mathcal{L}_{2}$')
        plt.yscale('log')
        plt.xlabel('LBFGS iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/LBFGS_loss.png')
        plt.show()
