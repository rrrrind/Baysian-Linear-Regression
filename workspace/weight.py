import numpy as np

class WeightDistribution(object):
    def __init__(self, term_num):
        self.prior_mean = np.zeros(term_num)
        self.prior_cov = np.ones([term_num,term_num])

    @property
    def prior_mean(self):
        return self.prior_mean

    @property
    def prior_cov(self):
        return self.prior_cov

    def _calc_posterior_cov(self, x_vec, get_y_sigma()):
        return (np.dot(x_vec, x_vec.T) / get_y_sigma()) + (1/self.prior_cov)

    def _calc_posterior_mean(self, posterior_cov, x_vec, y, get_y_sigma()):
        return posterior_cov * (((y@x_vec.T)/get_y_sigma()) + (1/self.prior_cov)@self.prior_mean)

    def update_params(self, x_vec, y, get_y_sigma()):
        posterior_cov = self._calc_weight_cov(x_vec, get_y_sigma())
        posterior_mean = self._calc_weight_mean(posterior_cov, x_vec, y, get_y_sigma())
        self.prior_cov = new_weight_cov
        self.prior_mean = new_weight_mean