import numpy as np
import preprocessing as pp

class NormalDistribution(object):
    def __init__(self, weight):
        self.weight = weight
        
        self.initial_sigma = 0.05 #今回は予測分布の分散を既知とする(ODの残差を参照)
        self.mean = None
        self.sigma = None
        
    @property
    def init_sigma(self):
        return self.initial_sigma
        
    @property
    def posterior_mean(self):
        return self.mean
    
    @property
    def posterior_sigma(self):
        return self.sigma
    
    def predict(self, x_scalar):
        self.mean = self._calc_posterior_mean(x_scalar)[0]
        self.sigma = self._calc_posterior_sigma(x_scalar)

    def _calc_posterior_mean(self, x_scalar):
        x_vec = pp.trans_scalar_to_vec(x_scalar, term_num=self.weight.term_num)
        return np.dot(self.weight.prior_mean.T, x_vec)
        
    def _calc_posterior_sigma(self, x_scalar):
        x_vec = pp.trans_scalar_to_vec(x_scalar, term_num=self.weight.term_num)
        return np.diag(self.initial_sigma + np.dot(np.dot(x_vec.T, self.weight.prior_cov), x_vec))
    