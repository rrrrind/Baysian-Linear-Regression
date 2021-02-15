from sys import argv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import visualization as vz
from dataset import ObservationData as OD
from weight import MultivariateNormalDistribution as mnd
from model import NormalDistribution as nd

OD_MEAN = float(argv[1]) # int or float
OD_VAR = float(argv[2])  # int or float
SAMPLING_NUM = int(argv[3]) # int
TERM_NUM = int(argv[4])     # int

def show_parameters():
    print("Mean of observation data : {}".format(OD_MEAN))
    print("Var of observation data  : {}".format(OD_VAR))
    print("Number of samples        : {}".format(SAMPLING_NUM))
    print("Number of terms          : {}".format(TERM_NUM))
    
def calc_variation(y_mean, y_var):
    y_min = y_mean - np.sqrt(y_var)
    y_max = y_mean + np.sqrt(y_var)
    return y_min, y_max

def train():
    show_parameters()
    
    od = OD(OD_MEAN, OD_VAR, SAMPLING_NUM)
    w_mnd = mnd(TERM_NUM)
    m_nd = nd(w_mnd)
    x_org, y_org = od.x_org, od.y_org

    for i in range(SAMPLING_NUM):
        w_mnd = mnd(TERM_NUM)
        for j in range(i+1):
            w_mnd.update_params(x_org[j], y_org[j], m_nd.init_sigma)

        m_nd = nd(w_mnd)
        m_nd.predict(od.x_range)
        
        y_mean = m_nd.posterior_mean
        y_min, y_max = calc_variation(y_mean, m_nd.posterior_sigma)
        
        vz.save_img(od.x_range, y_mean, y_min, y_max, x_org[:i+1], y_org[:i+1], i+1)
    
    vz.save_gif()

if __name__ == '__main__' :
    train()