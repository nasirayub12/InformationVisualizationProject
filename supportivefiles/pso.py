# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:30:25 2020

@author: nasir
"""
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

def PSO_Tuning(range):
    

    # Set-up hyperparameters
   # options = {'c1': 0.01, 'c2': 0.01, 'w':0.1}
    options = range;
    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=4, options=options)
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(fx.sphere, iters=200)
    return (best_cost)