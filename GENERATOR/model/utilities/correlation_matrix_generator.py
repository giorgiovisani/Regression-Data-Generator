import numpy as np
import pandas as pd
from scipy.stats import random_correlation
import numba
from numba import jit


class Correlation_Matrix_Generator():

    """ Static class that generates correlation matrix and standard deviations. Useful for many steps that require those generations """

    @staticmethod
    def gen_corr_matrix(seed, n_variables, related_vars, opt_level='float64'):
        
        """ 
        :param n_variables: number of features
        :param related_vars: number of related variables
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'.
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32;\n
        'float16' -> downcast to float16

        Generate a custom correlation matrix, with mixed correlated and incorrelated features
        """

        random_copy = np.random.RandomState(seed) #copia del seed
        np.random.seed(seed) # impostato per random_correlation
        
        def randomize_unrelated_position(random_copy, matrix, n_variables, related_vars):
            indice_massimo_inserimento = related_vars - 1 #indice ultima riga senza zeri
            indice_massimo = n_variables - 1 #indice ultima riga

            for _ in range(n_variables - related_vars):
                rand = random_copy.randint(0, indice_massimo)
                
                if(rand <= indice_massimo_inserimento):
                    index = indice_massimo 
                    for k in range(n_variables): 
                        while index >= rand :
                            if index > rand :
                                matrix[k, index] = matrix[k, index-1]
                            if index == rand :
                                matrix[k, index] = 0
                            index = index - 1
                        index = indice_massimo  
                    
                    index = indice_massimo
                    for k in range(n_variables):
                        while index >= rand :
                            if index > rand :
                                matrix[index, k] = matrix[index-1, k]
                            if index == rand :
                                matrix[index, k] = 0
                            index = index - 1
                        index = indice_massimo
                        
                    matrix[rand, rand] = 1
                    indice_massimo_inserimento = indice_massimo_inserimento + 1
            return matrix
        
        # eigenvalues (la somma è uguale alla dimensione della matrice)
        rand_nums = [random_copy.uniform(0.3,1.5) for _ in range(related_vars)]
        eigenvalues = []
        for num in rand_nums :
            eigenvalues.append((num/sum(rand_nums))*related_vars)
        
        correlation_matrix_part = random_correlation.rvs(eigenvalues) # based on np.random.seed
        
        correlation_matrix_zeros_high_indices = np.zeros((n_variables,)*2)
        np.fill_diagonal(correlation_matrix_zeros_high_indices, [1 for _ in range(n_variables)])
        correlation_matrix_zeros_high_indices[0:related_vars,0:related_vars] = correlation_matrix_part
        
        correlation_matrix_x = randomize_unrelated_position(random_copy, correlation_matrix_zeros_high_indices, n_variables, related_vars)
        
        return pd.DataFrame(correlation_matrix_x.astype(dtype='float64', copy=True))

    @staticmethod
    def gen_standard_devs(seed, n_variables, related_vars, n_scales, opt_level='float64'):

        """
        :param n_variables: number of features
        :param related_vars: number of related variables
        :param n_scales: number of scales
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'
        
        Notes
        ------
        
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16
        """


        variance_scales = [[5,10],[10,20],[100,200],[1000,2000]]
    
        #Calculate number of unrelated variables
        unrelated_vars = n_variables - related_vars
        
        #n_vars_scale: list with two lists, # of varaibles with the same scale 
        #(at half list you have to repeat (related first, unrelated second))
        n_vars_scale_inner = []
        for n in related_vars,unrelated_vars:
            n1,n2 = divmod(n,n_scales)
            n_vars_scale_inner.append([n1+n2 if pos==0 else n1 for pos in range(n_scales)])
        n_vars_scale = [val for sublist in n_vars_scale_inner for val in sublist]
        
        #Generate vector of standard deviations of the Variables
        stdevs = np.array([np.sqrt(round(np.random.uniform(variance_scales[pos][0],
                                                        variance_scales[pos][1]),1)) 
        for pos,rep in zip(list(range(int(len(n_vars_scale)/2)))*2,n_vars_scale) for _ in range(rep)])
        stdevs = np.array(stdevs)[np.newaxis]
        return stdevs.astype(dtype=opt_level, copy=False)      