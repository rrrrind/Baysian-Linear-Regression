import numpy as np
import preprocessing as pp

class MultivariateNormalDistribution(object):
    def __init__(self, term_num):
        self.term_num = term_num
        
        self.mean = np.ones([term_num, 1])
        self.cov = np.ones([term_num,term_num])
        self.acc = np.linalg.pinv(self.cov)

    @property
    def prior_mean(self):
        return self.mean

    @property
    def prior_cov(self):
        return self.cov

    def _calc_posterior_acc(self, x_org_scalar, initial_sigma):
        x_vec = pp.trans_scalar_to_vec(x_org_scalar, term_num=self.term_num)
        return ((1/initial_sigma) * np.dot(x_vec, x_vec.T)) + self.acc

    def _calc_posterior_mean(self, posterior_acc, x_org_scalar, y_org, initial_sigma):
        x_vec = pp.trans_scalar_to_vec(x_org_scalar, term_num=self.term_num)
        return np.dot(np.linalg.pinv(posterior_acc), 
                     (((1/initial_sigma)*y_org*x_vec)+np.dot(self.acc, self.mean)))

    def update_params(self, x_org_scalar, y_org, initial_sigma):
        posterior_acc = self._calc_posterior_acc(x_org_scalar, initial_sigma)
        posterior_mean = self._calc_posterior_mean(posterior_acc, x_org_scalar, y_org, initial_sigma)
        self.acc = posterior_acc
        self.cov = np.linalg.pinv(posterior_acc)
        self.mean = posterior_mean