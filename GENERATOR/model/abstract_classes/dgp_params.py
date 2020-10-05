from abc import ABC, abstractmethod # for abstract methods

class DGP_Params(ABC):
    """ An abstract class for DGP params """

    def __init__(self, seed=None, n_variables=None, error_variance=None, n_points=None):
        pass
        
    def validate_params(self, seed=None, n_variables=None, error_variance=None, n_points=None):
        pass