import numpy as np

def trans_scalar_to_vec(x_scalar, term_num):
    x_vec = np.zeros([term_num, len(x_scalar)])
    for i in range(term_num):
        for j, x_raw in enumerate(x_scalar):
            x_vec[i,j] = x_raw**(term_num-(i+1)) 
    return x_vec