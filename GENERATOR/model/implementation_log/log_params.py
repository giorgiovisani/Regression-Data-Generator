import numpy as np
import pandas as pd
from model.abstract_classes.dgp_params import DGP_Params

class Log_Params(DGP_Params):

    """ Inplementation of DGP_Params, used for Logistic Regression Data Generating Process """
    
    def __init__(self, seed=None, n_variables=None, error_variance=None, \
        n_points=None, related_vars=None, n_scales=None, requested_odds=None,unuseful_vars=None,betas=None):

        """
        :param seed: int, the seed for initialize random generation. If not set, will be randomized
        :param n_variables: int, number of variables
        :param error_variance: int, error variance of error vector
        :param n_points: int, it represents the number of points that will be generated
        :param related_vars: int, number of related variables
        :param n_scales: int, number of scales
        :param requested_odds: int, odd of vector Y
        :param unuseful_vars: int, number of 0 betas
        :param betas: array. If not set, it will be randomized considering unuseful_vars
        """
        
       
        self.validate_params_and_create_obj(seed, n_variables, error_variance, \
            n_points, related_vars, n_scales, requested_odds, unuseful_vars, betas)


    def validate_params_and_create_obj(self, seed=None, n_variables=None, error_variance=None, n_points=None, \
        related_vars=None, n_scales=None, requested_odds=None, unuseful_vars=None, betas=None):

        if seed is None: # seed is randomized
            # se il seed viene generato non e' possibile ricavarselo
            # è possibile invece riprendersi lo stato del seed:
            # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
            self.seed = np.random.seed()
        else:
            self.seed = seed
            
        # Set the seed
        self.initialize_randomization()
        
        random_copy = np.random.RandomState(self.seed) # mi faccio una copia, non voglio mai alterare lo stato globale

        if n_variables is None or n_variables <= 0:
            raise ValueError("Parameter 'n_variables' is not valid")
            sys.exit(1)
        else:
            self.n_variables = n_variables

        if error_variance is None: 
            raise ValueError("Parameter 'error_variance' is not valid")
            sys.exit(1)
        else:
            self.error_variance = error_variance   

        if n_points is None or n_points <= 0: 
            raise ValueError("Parameter 'n_points' is not valid")
            sys.exit(1)
        else:
            self.n_points = n_points

        if related_vars is None or related_vars < 0: 
            raise ValueError("Parameter 'related_vars' is not valid")
            sys.exit(1)
        elif related_vars >= self.n_variables:
            raise ValueError("Parameter 'related_vars' must be < than 'n_variables'")
            sys.exit(1)
        else:
            self.related_vars = related_vars      

        if n_scales is None or n_scales <= 0:
            raise ValueError("Parameter 'n_scales' is not valid")
            sys.exit(1)
        else:
            self.n_scales = n_scales

        # requested_odds represents the X odd
        if requested_odds is None: 
            raise ValueError("Parameter 'requested_odds' is not valid")
            sys.exit(1)
        else:
            self.requested_odds = requested_odds
            
        # gen X means if true
        self.means = [round(random_copy.uniform(0, 3), 1) for _ in range(n_variables)]
        self.means = list(np.multiply(self.means, [[-1, 1][random_copy.randint(2)] for _ in range(len(self.means))]))
        
        self.unuseful_vars = unuseful_vars
        
        if betas is None : # is randomized
            if unuseful_vars is None : 
                self.unuseful_vars = 0
            elif unuseful_vars <= 0: 
                raise ValueError("Parameter 'unuseful_vars' is not valid")
                sys.exit(1)
            elif unuseful_vars >= self.n_variables:
                raise ValueError("Parameter 'unuseful_vars' must be < than 'n_variables'")
                sys.exit(1)
            
            # Generate vector of coefficients (random[0,3]), randomly switch some to negative
            betas = [round(random_copy.uniform(0.5, 2), 1) for _ in range(n_variables - unuseful_vars)]
            betas = list(np.multiply(betas, [[-1, 1][random_copy.randint(2)] for _ in range(len(betas))]))
            
            # Add such zeros as the unuseful variables in random positions of the beta vector
            for _ in range(unuseful_vars):
                index = np.random.randint(0, len(betas) + 1)
                betas.insert(index, 0.0)

            # Voglio controllare il valore medio (solo la parte deterministica X*beta) nel nostro campione.
            intercept_value = round(np.log(requested_odds) - sum([a * c for a, c in zip(betas, self.means[:len(betas)])]), 2)

            betas.insert(0, intercept_value)
        
        self.betas = pd.Series(betas) # create pandas object


    def initialize_randomization(self):
        np.random.RandomState(self.seed) # uso il seed per generare il primo stato. Gli altri saranno generati automaticamente
        np.random.seed(self.seed)