# [depends] %LIB%/ABC_plant.py
# [depends] ABC_solve.pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.linalg import expm
from scipy.linalg import lstsq

figsize = (4,3)
data = pickle.load(open('ABC_solve.pickle', 'rb'))
t = data['t']
y_meas = data['y_meas']
y_pred = data['y_pred']
y_pred_res = data['y_pred_res']
y = data['y']
y[:,:,3] = np.exp(y[:,:,3])

with PdfPages('ABC_plot.pdf') as pdf:

    plt.figure(figsize=figsize)
    plt.plot(t, y_meas[:,0], 'o', color='red', mfc='none')
    plt.plot(t, y_pred[:,0], '-', color='red')
    plt.plot(t, y_meas[:,1], 'o', color='blue', mfc='none')
    plt.plot(t, y_pred[:,1], '-', color='blue')
    plt.plot(t, y_meas[:,2], 'o', color='green', mfc='none')
    plt.plot(t, y_pred[:,2], '-', color='green')
    plt.xlabel('$t$')
    plt.ylabel('$C$', rotation=0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=figsize)
    plt.plot(t, y_meas[:,0], 'o', color='red', mfc='none')
    plt.plot(t, y_pred_res[:,0], '-', color='red')
    plt.plot(t, y_meas[:,1], 'o', color='blue', mfc='none')
    plt.plot(t, y_pred_res[:,1], '-', color='blue')
    plt.plot(t, y_meas[:,2], 'o', color='green', mfc='none')
    plt.plot(t, y_pred_res[:,2], '-', color='green')
    plt.xlabel('$t$')
    plt.ylabel('$C$', rotation=0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    T, _ = np.meshgrid(t, np.arange(y.shape[1]), indexing='ij')  # (n_timesteps, n_samples)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), subplot_kw={"projection": "3d"})  # 3D subplots
    label = {'0': '$c_A$', '1': '$c_B$', '2': '$c_C$'}
    for i, ax in enumerate(axes):
        Y = y[:, :, i]  # Y-values (n_timesteps, n_samples)
        Z = y[:, :, 3]  # Z-values (n_timesteps, n_samples)
        
        # Wireframe plot (mesh without color fill)
        ax.plot_wireframe(T, Y, Z, color='blue', linewidth=0.5)
        #ax.contourf(T, Y, Z, zdir='x', offset=0, cmap='coolwarm', alpha=0.5)
        ax.view_init(azim=20)
        # Label axes
        ax.set_xlabel('$t$')
        ax.set_ylabel(f'{label[str(i)]}')
        if i == 0:
            ax.set_zlabel('$p(c, t)$')
    # The fix
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.99, wspace=0.05, hspace=0.1)
    pdf.savefig()
    plt.close()

    par = 10
    plt.figure(figsize=figsize)
    t = np.linspace(0, 1, 100)
    exppos = np.exp(par*t)
    expneg = np.exp(-par*t)
    t = np.linspace(1, 0, 100)
    plt.semilogy(t, exppos, '-', color='red', label='Exploding grad')
    plt.semilogy(t, expneg, '-', color='blue', label='Vanishing grad')
    plt.xlabel('$t$')
    plt.ylabel('$\lambda$', rotation=0)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # K = np.array([[-1, 1/3], [1, -(1/3+0.1)]])
    # t = np.linspace(0, 50, 200)
    # eps = 0.1
    # c0 = np.array([1 + eps, 0 + eps])
    # c0_perturb = np.array([1 + 2*eps, 0 + 2*eps])
    # c0_perturb2 = np.array([1 + 3*eps, 0 + 3*eps])
    # c = np.array([c0 @ expm(K*t) for t in t])
    # c_perturb = np.array([c0_perturb @ expm(K*t) for t in t])
    # c_perturb2 = np.array([c0_perturb2 @ expm(K*t) for t in t])

    # plt.figure(figsize=figsize)
    # plt.plot(t, c[:,1], '-', color='red', label='$c_{B1}$')
    # plt.plot(t, c_perturb[:,1], '--', color='blue', label='$c_{B2}$')
    # plt.plot(t, c_perturb2[:,1], color='green', label='$c_{B3}$')
    # plt.plot(t[:-2], c[2:,1], color='black', label='$c_{B1}(t + T)$, T=0.5')

    # plt.xlabel('$t$')
    # plt.ylabel('$c_B$', rotation=0)
    # plt.legend()
    # plt.tight_layout()
    # pdf.savefig()
    # plt.close()

    K = np.array([-1])
    t = np.linspace(0, 2, 75)
    eps = 0.1
    c0 = np.array([1 + eps])
    c0_perturb = np.array([1 + 2*eps])
    c0_perturb2 = np.array([1 + 3*eps])
    c = np.array([c0 @ expm(K*t) for t in t])
    c_perturb = np.array([c0_perturb @ expm(K*t) for t in t])
    c_perturb2 = np.array([c0_perturb2 @ expm(K*t) for t in t])

    plt.figure(figsize=figsize)
    plt.plot(t, c, '-', color='red', label='$c_{B1}$')
    plt.plot(t, c_perturb, '--', color='blue', label='$c_{B2}$')
    plt.plot(t, c_perturb2, color='green', label='$c_{B3}$')
    plt.plot(t[:-2], c[2:], 'o', color='black',ms=0.5, label='$c_{B1}(t + T)$, T=0.5')

    plt.xlabel('$t$')
    plt.ylabel('$c_B$', rotation=0)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    np.random.seed(42)
    std = 0.1
    x = np.linspace(-3, 3, 10) 
    y_true = np.sin(x) 
    y_noisy = y_true + np.random.normal(scale=0.1, size=x.shape) 
    degree = 10

    X = np.vander(x, degree + 1, increasing=True)
    coef_no_reg, _, _, _ = lstsq(X, y_noisy)

    D_derivative = np.zeros_like(X)  # Initialize to match the shape of X
    for j in range(1, degree + 1):  # Start from 1 since 0th derivative is ignored
        D_derivative[:, j] = j * x ** (j - 1) 
    lambda_derivative = 0.1  # Regularization strength for derivative penalty
    coef_derivative_reg = np.linalg.solve(X.T @ X + lambda_derivative * D_derivative.T @ D_derivative, X.T @ y_noisy)

    x_fine = np.linspace(-3, 3, 200)
    X_fine = np.vander(x_fine, degree + 1, increasing=True)
    y_pred_no_reg = X_fine @ coef_no_reg
    y_pred_reg = X_fine @ coef_derivative_reg

    # Plot results
    plt.figure(figsize=figsize)
    plt.plot(x, y_noisy,'ko', label="Noisy data", ms=3, mfc='none')
    plt.plot(x_fine, np.sin(x_fine), label="True function", linestyle="dashed", color="gray")
    plt.plot(x_fine, y_pred_no_reg, label="Overfit", color="red")
    plt.plot(x_fine, y_pred_reg, label="Regularized", color="blue")

    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()