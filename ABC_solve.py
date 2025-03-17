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

# generate data for CNF
abc.multi_solve(ntimes=10)
# train the CNF
abc.cnf_train(niter=500)

data = {}
data['t'] = t
data['y_meas'] = y_meas
data['y_pred'] = y_pred
data['y'] = abc.y

with open('ABC_solve.pickle', 'wb') as f:
    pickle.dump(data, f)


