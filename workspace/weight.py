import numpy as np
import preprocessing as pp

class MultivariateNormalDistribution(object):
    def __init__(self, term_num):
        self.term_num = term_num
        
        self.prior_mean = np.zeros(term_num)
        self.prior_cov = np.ones([term_num,term_num])

    @property
    def prior_mean(self):
        return self.prior_mean

    @property
    def prior_cov(self):
        return self.prior_cov

    def _calc_posterior_cov(self, x_org_scalar, initial_sigma):
        x_vec = pp.trans_scalar_to_vec(x_org_scalar, term_num=self.term_num)
        return (np.dot(x_vec, x_vec.T) / initial_sigma) + (1/self.prior_cov)

    def _calc_posterior_mean(self, posterior_cov, x_org_scalar, y_org, initial_sigma):
        x_vec = pp.trans_scalar_to_vec(x_org_scalar, term_num=self.term_num)
        return posterior_cov * (((y@x_vec.T)/initial_sigma) + (1/self.prior_cov)@self.prior_mean)

    def update_params(self, x_org_scalar, y_org, initial_sigma):
        posterior_cov = self._calc_posterior_cov(x_org_scalar, initial_sigma)
        posterior_mean = self._calc_posterior_mean(posterior_cov, x_org_scalar, y_org, initial_sigma)
        self.prior_cov = posterior_cov
        self.prior_mean = posterior_mean