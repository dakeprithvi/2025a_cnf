# [depends] %LIB%/ABC_plant.py

import sys
sys.path.append('lib/')
from ABC_plant import ABC 
import pickle

abc = ABC()
# get measurements
t, y_meas = abc.measurement()
# train the model
abc.train(niter=500)
# get predictions
_, y_pred = abc.lsim(use_measure_y0=True)

abc.resnet_train(niter=500)
# get predictions
_, y_pred_res = abc.resnet_lsim(use_measure_y0=True)


# generate data for CNF
abc.multi_solve(ntimes=10)
# train the CNF
abc.cnf_train(niter=500)


abc.pnn_train()

data = {}
data['t'] = t
data['y_meas'] = y_meas
data['y_pred'] = y_pred
data['y_pred_res'] = y_pred_res
data['y'] = abc.y
data['y_pnn'] = abc.c_pred

with open('ABC_solve.pickle', 'wb') as f:
    pickle.dump(data, f)


