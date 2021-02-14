import numpy as np
import preprocessing as pp

class NormalDistribution(object):
    def __init__(self, weight, x_range):
        self.weight = weight
        self.x_range = x_range
        
        self.initial_sigma = 1 #今回は予測分布の分散を既知とする
        self.prior_mean = np.zeros(len(x_range)) 
        self.prior_sigma = np.ones(len(x_range))
        
        self._calc_prior_mean() # 初期の期待値(y_pred)の計算
        
    @property
    def initial_sigma(self):
        return self.initial_sigma
        
    @property
    def prior_mean(self):
        return self.prior_mean
    
    @property
    def prior_sigma(self):
        return self.prior_sigma
    
    def update_params(self):
        self.prior_mean = self._calc_posterior_mean()
        self.prior_sigma = self._calc_posterior_sigma()
    
    def predict(self, x_scalar):
        x_vec = pp.trans_scalar_to_vec(x_scalar, term_num=self.weight.term_num)
        return np.dot(self.weight.prior_mean, x_vec)[0]

    def _calc_posterior_mean(self):
        x_vec = pp.trans_scalar_to_vec(self.x_range, term_num=self.weight.term_num)
        self.prior_mean = np.dot(self.weight.prior_mean, x_vec)[0]
        
    def _calc_posterior_sigma(self):
        x_vec = pp.trans_scalar_to_vec(self.x_range, term_num=self.weight.term_num)
        self.prior_sigma = self.initial_sigma + np.dot(np.dot(x_vec.T, self.weight.prior_cov)[0], x_vec)[0]
    