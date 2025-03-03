import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import time

class ABC(nn.Module):
    def __init__(self, config=None):
        super(ABC, self).__init__()
        if config is None:
            config = {}
        self.k1 = 1
        self.k_1 = 1/3
        self.k2 = 0.1
        self.ca0 = 1
        self.cb0 = 0
        self.cc0 = 0
        self.t_start = 0
        self.t_end = 6
        self.nplot = 75
        self.dtype = torch.float32
        self.mean = 0
        self.std = 0.05
        self.std_c = torch.tensor([0.05] * 3)
        self.plant_solve = True
        self.nn =  nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )
        self.nn.apply(self.init_weights)

     
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -1e-3, 1e-3).float()

    def rate_true(self, y):
        ca, cb, cc = y
        r1 = self.k1 * ca - self.k_1 * cb
        r2 = self.k2 * cb
        return r1, r2
    
    def rate_nn(self, y):
        r = self.nn(y)
        return r

    def plant(self, t, y):
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
    
    def solve_plant(self):
        self.y0 = torch.tensor([self.ca0, self.cb0, self.cc0], dtype = self.dtype)
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype = self.dtype)
        y = torchdiffeq.odeint(self.plant, self.y0, t, method = 'dopri5')
        return t, y

    def measurement(self):
        t, y = self.solve_plant()
        y = y + torch.normal(mean = self.mean, std = self.std, size = y.size())
        self.measure = y
        return t.numpy(), y.numpy()
    
    def loss(self):
        self.plant_solve = False
        _, y_pred = self.solve_plant()
        loss = torch.mean((y_pred - self.measure)**2)
        return loss
    
    def train(self, niter = 1000):
        optimizer = optim.Adam(self.nn.parameters())
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

    def lsim(self):
        self.plant_solve = False
        y0 = torch.tensor([self.ca0, self.cb0, self.cc0], dtype = self.dtype)
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype = self.dtype)
        y = torchdiffeq.odeint(self.plant, y0, t, method = 'dopri5')
        return t.detach().numpy(), y.detach().numpy()

    def multi_solve(self, ntimes = 100):
        y0 = torch.normal(mean = self.y0.expand(ntimes, -1), std = self.std_c.expand(ntimes, -1))
        self.y0_multi = y0
    

abc = ABC()
t, y_meas = abc.measurement()
abc.train(niter=50)
t, y_pred = abc.lsim()

plt.plot(t, y_meas, 'o')
plt.plot(t, y_pred, '-')
plt.show()

abc.multi_solve(ntimes=1000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(abc.y0_multi[:, 0], abc.y0_multi[:, 1], abc.y0_multi[:, 2], 'o')
plt.show()