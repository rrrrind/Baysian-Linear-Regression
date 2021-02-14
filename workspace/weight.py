import numpy as np
import preprocessing as pp

class MultivariateNormalDistribution(object):
    def __init__(self, term_num):
        self.term_num = term_num
        
        self.prior_mean = np.ones([term_num, 1])
        self.prior_cov = np.ones([term_num,term_num])
        self.prior_acc = np.linalg.pinv(self.prior_cov)

    @property
    def mean(self):
        return self.prior_mean

    @property
    def cov(self):
        return self.prior_cov

    def _calc_posterior_acc(self, x_org_scalar, initial_sigma):
        x_vec = pp.trans_scalar_to_vec(x_org_scalar, term_num=self.term_num)
        return (initial_sigma * np.dot(x_vec, x_vec.T)) + self.prior_acc

    def _calc_posterior_mean(self, posterior_acc, x_org_scalar, y_org, initial_sigma):
        x_vec = pp.trans_scalar_to_vec(x_org_scalar, term_num=self.term_num)
        return np.dot(np.linalg.pinv(posterior_acc), 
                     ((initial_sigma*y_org*x_vec)+np.dot(self.prior_acc, self.prior_mean)))

    def update_params(self, x_org_scalar, y_org, initial_sigma):
        posterior_acc = self._calc_posterior_acc(x_org_scalar, initial_sigma)
        posterior_mean = self._calc_posterior_mean(posterior_acc, x_org_scalar, y_org, initial_sigma)
        self.prior_acc = posterior_acc
        self.prior_cov = np.linalg.inv(posterior_acc)
        self.prior_mean = posterior_mean