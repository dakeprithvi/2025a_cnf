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
        super(ABC, self).__init__()
        if config is None:
            config = {}
        self.k1 = 1
        self.k_1 = 1/3
        self.k2 = 0.1
        self.ca0 = 4
        self.cb0 = 1
        self.cc0 = 1
        self.t_start = 0
        self.t_end = 6
        self.nplot = 75
        self.dtype = torch.float32
        self.mean = 0
        self.std = 0.1
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
        self.plant_solve = True
        self.y0_multi = y0.detach()
        self.ca0_multi = np.zeros((ntimes, self.nplot))
        self.cb0_multi = np.zeros((ntimes, self.nplot))
        self.cc0_multi = np.zeros((ntimes, self.nplot))
        self.y0_pdf = torch.tensor(stats.multivariate_normal.pdf(self.y0_multi, mean = self.y0, cov = self.std_c * np.eye(self.y0.shape[0])), dtype = self.dtype)
        for i in range(ntimes):
            self.y0 = self.y0_multi[i, :]
            _, y = self.measurement()
            self.ca0_multi[i, :] = y[:, 0]
            self.cb0_multi[i, :] = y[:, 1]
            self.cc0_multi[i, :] = y[:, 2]

        axis=0
        mean_y0 = self.y0_multi.mean(axis=axis)
        std_y0 = self.y0_multi.std(axis=axis)
        self.y0_norm_multi = (self.y0_multi - mean_y0) / std_y0
        self.ca0_norm_multi = torch.tensor((self.ca0_multi - self.ca0_multi.mean(axis=axis)) / self.ca0_multi.std(axis=axis), dtype = self.dtype)
        self.cb0_norm_multi = torch.tensor((self.cb0_multi - self.cb0_multi.mean(axis=axis)) / self.cb0_multi.std(axis=axis), dtype = self.dtype)
        self.cc0_norm_multi = torch.tensor((self.cc0_multi - self.cc0_multi.mean(axis=axis)) / self.cc0_multi.std(axis=axis), dtype = self.dtype)
        self.y0_cnf = torch.cat((self.y0_norm_multi, self.y0_pdf.reshape(-1,1)), axis=1)
        self.cnf =  nn.Sequential(
            nn.Linear(3 , 8),
            nn.Tanh(),
            nn.Linear(8, 3)
        )
        self.cnf.apply(self.init_weights)

    def cn_solve(self, t, y):
        r = self.cnf(y[:, :3])
        print(y.shape)
        trace = torch.zeros(y.shape[0], 1)
        for i in range(y.shape[0]):
            grad = torch.autograd.functional.jacobian(self.cnf, y[i, :3], create_graph=True)
            trace[i] = torch.sum(torch.diagonal(grad))
        print(r.shape, trace.shape)
        r = torch.cat((r, trace), axis=1)
        return r
    
    def cn_loss(self):
        t = torch.linspace(self.t_start, self.t_end, self.nplot, dtype = self.dtype)
        self.y = torchdiffeq.odeint(self.cn_solve, self.y0_cnf, t, method = 'dopri5')
        # print(y.shape)
        # self.grad = torch.autograd.functional.jacobian(self.cnf, self.y[-1,-1,:].T, create_graph=True)
        # self.trace = torch.sum(torch.diagonal(self.grad, dim1=-2, dim2=-1))
        # print(self.trace.shape)
        # self.initial = torch.log(self.y0_norm_multi) + self.trace * (self.t_end - self.t_start)
        # self.final = torch.log(self.y[-1, :, :])
        #loss = torch.mean((self.initial - self.final)**2)
    

abc = ABC()
t, y_meas = abc.measurement()
abc.train(niter=50)
t, y_pred = abc.lsim()

plt.plot(t, y_meas, 'o')
plt.plot(t, y_pred, '-')

abc.multi_solve(ntimes=10)
ind = -1
x0 = abc.ca0_multi[:, 0]
y0 = abc.cb0_multi[:, 0]
z0 = abc.cc0_multi[:, 0]
x = abc.ca0_multi[:, ind]
y = abc.cb0_multi[:, ind]
z = abc.cc0_multi[:, ind]

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

# Create 3D plot (center plot)
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.scatter(x, y, z, 'o')
ax1.scatter(x0, y0, z0, 'o')
ax1.set_title('3D Scatter Plot')
ax1.set_xlabel('$c_A$')
ax1.set_ylabel('$c_B$')
ax1.set_zlabel('$c_C$')
ax1.xaxis.pane.fill = False  # Remove XY background
ax1.yaxis.pane.fill = False  # Remove YZ background
ax1.zaxis.pane.fill = False  # Remove ZX background
ax1.view_init(azim=60)

cmap = 'Blues'
bins = 20
# Compute 2D histogram
hist, yedges, zedges = np.histogram2d(y, z, bins=bins)

# Convert to interpolated image
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(hist.T, origin='lower', cmap=cmap, 
                interpolation='bicubic',  # Try 'gaussian' or 'bicubic'
                extent=[yedges[0], yedges[-1], zedges[0], zedges[-1]],
                aspect='auto')

# Labels
ax2.set_xlabel('$c_B$')
ax2.set_ylabel('$c_C$', rotation=0)

# Create 2D heatmap in the xy-plane (bottom plot)
ax3 = fig.add_subplot(gs[1, 0])
# Compute 2D histogram
hist, yedges, zedges = np.histogram2d(x, y, bins=bins)
im = ax3.imshow(hist.T, origin='lower', cmap=cmap, 
                interpolation='bicubic',  # Try 'gaussian' or 'bicubic'
                extent=[yedges[0], yedges[-1], zedges[0], zedges[-1]],
                aspect='auto')

# Labels
ax3.set_xlabel('$c_A$')
ax3.set_ylabel('$c_B$', rotation=0)

# Adjust layout to prevent overlap
plt.tight_layout(pad=5)

# Show the plot

plt.savefig('ABC_deter.pdf')