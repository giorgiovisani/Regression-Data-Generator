from abc import ABC, abstractmethod # for abstract methods

class DGP_Results(ABC):
    """ An abstract class for DGP Results """

    def __init__(self, features=None, response=None):
        pass
        
    def validate_results(self, features=None, response=None):
        pass