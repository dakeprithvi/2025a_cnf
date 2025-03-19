# [depends] %LIB%/ABC_plant.py
# [depends] ABC_solve.pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle

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
    plt.show()
    plt.close()
