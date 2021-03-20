
import numpy as np

PGD_SEARCH_RANGE = ['local', 'l2', [36/255., 64/255., 0.3, 0.5, 0.7, 0.9, 1.1], None, 40]
CW_SEARCH_RANGE = ['local', 'l2', 1.0, 1e-2, 100]

PARAMETERS = {
    "batch_size":32,
    "data_min": np.min((0 - np.array([0.485, 0.456, 0.406])) /
                      np.array([0.229, 0.224, 0.225])),
    "data_max": np.max((1 - np.array([0.485, 0.456, 0.406])) /
                      np.array([0.229, 0.224, 0.225])),
    "device": "cuda:0",
    "n_steps":10,
    "stdevs":0.15,
}

PIPELINE = {
    "methods": "PGDs+CW",
    "norm" : "l2",
    "pgd_eps": [36/255., 64/255., 0.3, 0.5, 0.7, 0.9, 1.1],
    "pgd_step_size": None,
    "pgd_max_steps": 40,
    "cw_eps": 1.0,
    "cw_step_size": 1.e-2,
    "cw_max_steps": 100,
}