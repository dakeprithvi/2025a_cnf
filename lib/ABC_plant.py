# Author: Prithvi Dake
# Course: ME255NN

import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

class ABC(nn.Module):
    def __init__(self, config=None):
        '''
        We study following kinetics: A <-> B -> C
        with rate constants k1, k_1, k2. The code fits a neural ODE and 
        also attempts to perform forward supervised CNF (which is not quite the
        same as the original CNF proposed by Chen et al. 2018).
        Link: https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf

        Maybe I can call my model a generative model.
        '''
        super(ABC, self).__init__()
        if config is None:
            config = {}
        # Default parameters
        self.k1 = 1
        self.k_1 = 1/3
        self.k2 = 0.1
        self.ca0 = 4
        self.cb0 = 1
        self.cc0 = 1

        # Trajectory start, end and number of points
        self.t_start = 0
        self.t_end = 6
        self.nplot = 75

        # Data type
        self.dtype = torch.float32
        self.method = 'euler'
        # Can try: 'euler', 'rk4', 'dopri5'

        # Noise parameters
        self.mean = 0
        self.std = 0.1
        self.std_c = torch.tensor([self.std * 2.5] * 3)
        self.plant_solve = True

        # Neural ODE
        self.nn =  nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )
        self.nn.apply(self.init_weights)

        self.resnet = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 3)
        )
        self.resnet.apply(self.init_weights)

        # CNF
        self.cnf =  nn.Sequential(
            nn.Linear(3 , 8),
            nn.Tanh(),
            nn.Linear(8, 3)
        )
        self.cnf.apply(self.init_weights)

        self.pnn = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 3)
        )
        self.pnn.apply(self.init_weights)

     
    @staticmethod
    def init_weights(m):
        '''
        Initialize weights of the neural network
        '''
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -1e-3, 1e-3).float()

    def rate_true(self, y):
        '''
        True rate equations
        '''
        ca, cb, cc = y
        r1 = self.k1 * ca - self.k_1 * cb
        r2 = self.k2 * cb
        return r1, r2
    
    def rate_nn(self, y):
        '''
        Neural network rate equations
        '''
        r = self.nn(y)
        return r

    def plant(self, t, y):
        '''
        The plant method defines the model using either true or neural network 
        rate equations.
        '''
        ca, cb, cc = y
        if self.plant_solve:
            r1, r2 = self.rate_true(y)
        else:
            r = self.rate_nn(y)
            r1 = r[0]
            r2 = r[1]
        dca_dt = -r1
        dcb_dt = 2 * r1 - r2
        dcc_dt = r2
        return torch.stack([dca_dt, dcb_dt, dcc_dt])
    
    def solve_plant(self, use_measure_y0=False, y0=None):
        '''
        Solve the plant model, take care to use the measured initial conditions,
        for the neural network since it is trained on the measured data.
        '''
        if use_measure_y0:
            yt0 = self.measure[0, :]
        elif y0 is not None:
            yt0 = y0
        else:
            yt0 = torch.tensor([self.ca0, self.cb0, self.cc0], dtype = self.dtype)
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype = self.dtype)
        y = torchdiffeq.odeint(self.plant, yt0, t, method = self.method)
        return t, y

    def measurement(self, y0=None):
        '''
        Generate measurements with noise for true rate equations
        '''
        self.plant_solve = True
        t, y = self.solve_plant(y0=y0)
        y = y + torch.normal(mean = self.mean, std = self.std, size = y.size())
        self.measure = y
        return t.numpy(), y.numpy()
    
    def loss(self):
        '''
        Get the loss to train the neural network
        '''
        self.plant_solve = False
        _, y_pred = self.solve_plant(use_measure_y0=True)
        loss = torch.mean((y_pred - self.measure)**2)
        return loss
    
    def train(self, niter = 1000):
        '''
        Train the neural network
        '''
        optimizer = optim.Adam(self.nn.parameters(), lr=1e-2)
        tic = time.time()
        for i in range(niter):
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss {loss.item()}')
        toc = time.time()
        print(f'Training time: {(toc - tic)/60} mins')


    def resnet_plant(self, use_measure_y0=False, y0=None):
        '''
        Get the loss to train the resnet
        '''
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype = self.dtype)
        if use_measure_y0:
            yt0 = self.measure[0, :]
        elif y0 is not None:
            yt0 = y0
        else:
            yt0 = torch.tensor([self.ca0, self.cb0, self.cc0], dtype = self.dtype)
        
        y_sol = [yt0]
        for i in range(len(t) - 1):
            next_y = self.resnet(y_sol[-1]) + y_sol[-1]
            y_sol.append(next_y)
        y_sol = torch.stack(y_sol)
        #loss = torch.mean((y_sol - self.measure)**2)
        return t, y_sol
    
    def resnet_loss(self, use_measure_y0=False):
        '''
        Get the loss to train the resnet
        '''
        t, y_sol = self.resnet_plant(use_measure_y0=use_measure_y0)
        loss = torch.mean((y_sol - self.measure)**2)
        return loss
    
    def resnet_train(self, niter = 1000):
        '''
        Train the resnet
        '''
        optimizer = optim.Adam(self.resnet.parameters(), lr=1e-2)
        tic = time.time()
        for i in range(niter):
            optimizer.zero_grad()
            loss = self.resnet_loss(use_measure_y0=True)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss {loss.item()}')
        toc = time.time()
        print(f'Training time: {(toc - tic)/60} mins')

    def resnet_lsim(self, use_measure_y0=False, y0=None):
        '''
        Mimic matlab's lsim: just re-solves the fitted model and returns the
        solution. Take care to use the measured initial conditions for the neural
        network since it is trained on the measured data.
        '''
        t, y_sol = self.resnet_plant(use_measure_y0=True)
        return t.detach().numpy(), y_sol.detach().numpy()

    def lsim(self, use_measure_y0=False, y0=None):
        '''
        Mimic matlab's lsim: just re-solves the fitted model and returns the 
        solution. Take care to use the measured initial conditions for the neural
        network since it is trained on the measured data.
        '''
        self.plant_solve = False
        if use_measure_y0:
            yt0 = self.measure[0, :]
        elif y0 is not None:
            yt0 = y0
        else:
            yt0 = torch.tensor([self.ca0, self.cb0, self.cc0], dtype = self.dtype)
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype = self.dtype)
        y = torchdiffeq.odeint(self.plant, yt0, t, method = self.method)
        return t.detach().numpy(), y.detach().numpy()

    def multi_solve(self, ntimes = 10):
        '''
        Simulate the data set for the CNF training
        '''
        yt0 = torch.tensor([self.ca0, self.cb0, self.cc0], dtype = self.dtype)
        y0 = torch.normal(mean = yt0.expand(ntimes, -1), std = self.std_c.expand(ntimes, -1))
        self.plant_solve = True
        self.y0_multi = y0.detach()
        self.ca_multi = np.zeros((ntimes, self.nplot))
        self.cb_multi = np.zeros((ntimes, self.nplot))
        self.cc_multi = np.zeros((ntimes, self.nplot))
        self.y0_pdf = torch.tensor(stats.multivariate_normal.pdf(self.y0_multi, 
                                                                 mean = yt0,
                                                                 cov = (self.std_c**2) * np.eye(yt0.shape[0])),
                                                                 dtype = self.dtype)
        self.log_y0_pdf = torch.log(self.y0_pdf)
        for i in range(ntimes):
            y0 = self.y0_multi[i, :]
            _, y = self.measurement(y0=y0)
            self.ca_multi[i, :] = y[:, 0]
            self.cb_multi[i, :] = y[:, 1]
            self.cc_multi[i, :] = y[:, 2]

        self.y0_cnf = torch.cat((self.y0_multi, self.log_y0_pdf.reshape(-1,1)), axis=1)
        

    def cnf_solve(self, t, y):
        '''
        Solve the CNF model using the neural network
        '''
        r = self.cnf(y[:, :3])
        trace = torch.zeros(y.shape[0], 1)

        # Not the best way to get the trace of the Jacobian
        # Later work by Chen et al propose using hutchinson's trace estimator
        for i in range(y.shape[0]):
            grad = torch.autograd.functional.jacobian(self.cnf, y[i, :3], create_graph=True)
            trace[i] = torch.sum(torch.diagonal(grad))
        r = torch.cat((r, trace), axis=1)
        return r
    
    def cnf_loss(self):
        '''
        Get the loss to train the CNF. 
        Here I minimize the prediction error and
        also the maximize negative log-likelihood.
        '''
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype = self.dtype)
        self.y = torchdiffeq.odeint(self.cnf_solve, self.y0_cnf, t, method = self.method)
        self.pzt = self.y[-1, :, -1]
        self.ca = self.y[:, :, 0].T
        self.cb = self.y[:, :, 1].T
        self.cc = self.y[:, :, 2].T
        cal = torch.mean((self.ca - torch.tensor(self.ca_multi)) ** 2)
        cbl = torch.mean((self.cb - torch.tensor(self.cb_multi)) ** 2)
        ccl = torch.mean((self.cc - torch.tensor(self.cc_multi)) ** 2)
        self.ode_loss = cal + cbl + ccl
        self.loss = -torch.mean(self.pzt) + 5e1 * self.ode_loss
        return self.loss

    def cnf_train(self, niter = 1000):
        '''
        Train the CNF
        '''
        optimizer = optim.Adam(self.cnf.parameters(),lr=1e-2)
        tic = time.time()
        for i in range(niter):
            optimizer.zero_grad()
            loss = self.cnf_loss()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss {loss.item()}')
        toc = time.time()
        print(f'Training time: {(toc - tic)/60} mins')
        self.y = self.y.detach().numpy()

    
    def pnn_loss(self):
        '''
        Compute the PINN loss for fitting the kinetics ODE.
        '''
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype=self.dtype, requires_grad=True)

        c_pred = self.pnn(t.reshape(-1,1))
        c_a, c_b, c_c = c_pred[:, 0], c_pred[:, 1], c_pred[:, 2]

        dca_dt = torch.autograd.grad(c_a, t, grad_outputs=torch.ones_like(c_a), create_graph=True)[0]
        dcb_dt = torch.autograd.grad(c_b, t, grad_outputs=torch.ones_like(c_b), create_graph=True)[0]
        dcc_dt = torch.autograd.grad(c_c, t, grad_outputs=torch.ones_like(c_c), create_graph=True)[0]

        ode1 = torch.mean((dca_dt - (-self.k1 * c_a + self.k_1 * c_b))**2)
        ode2 = torch.mean((dcb_dt - (2 * self.k1 * c_a - (2 * self.k_1 + self.k2) * c_b))**2)  
        ode3 = torch.mean((dcc_dt - (self.k2 * c_b))**2)

        pred_loss = torch.mean((c_pred - self.measure)**2)
        self.loss_for_pnn = ode1 + ode2 + ode3 + pred_loss
        self.c_pred = c_pred
        return self.loss_for_pnn
    
    def pnn_train(self, niter = 1000):
        '''
        Train the PINN
        '''
        optimizer = optim.Adam(self.pnn.parameters(), lr=1e-2)
        tic = time.time()
        for i in range(niter):
            optimizer.zero_grad()
            loss = self.pnn_loss()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss {loss.item()}')
        toc = time.time()
        print(f'Training time: {(toc - tic)/60} mins')
        self.c_pred = self.c_pred.detach().numpy()


    

