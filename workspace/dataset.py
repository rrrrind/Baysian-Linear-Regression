import numpy as np

class ObservationData(object):
    def __init__(self, mu, std, sampling):
        np.random.seed(3)
        self.mu = mu
        self.std = std
        self.sampling = sampling
    
    def generate(self):
        x_scalar = np.random.normal(self.mu, self.std, self.sampling)
        # 正規分布
        y = ((np.exp((-(x_scalar - self.mu)**2)/(2*(self.std**2))) / (np.sqrt(2*np.pi)*self.std))
          + 0.05*np.random.randn(len(x_scalar)))
        return x_scalar, y