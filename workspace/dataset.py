import numpy as np

class ObservationData(object):
    def __init__(self, mu, var, sampling):
        self.x = None
        self.y = None
        self.generate(mu, var, sampling)
    
    def generate(self, mu, var, sampling):
        # 正規分布に従いサンプリング
        self.x = np.random.normal(mu, np.sqrt(var), sampling)
        # 残差
        residual = 0.05 * np.random.randn(len(self.x))
        # 正規分布 + 残差
        self.y = ((np.exp((-(self.x - mu)**2)/(2*(std**2))) / (np.sqrt(2*np.pi)*std)) + residual)
        
    @property
    def x_org(self):
        return self.x
    
    @property
    def y_org(self):
        return self.y
    
    @property
    def x_range(self):
        return np.arange(np.min(self.x)-0.5, np.max(self.x)+0.5, 0.02)