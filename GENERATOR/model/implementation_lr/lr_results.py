import pandas as pd
from model.abstract_classes.dgp_results import DGP_Results

class LR_Results(DGP_Results):

    """ Inplementation of DGP_Results, used for Linear Regression Data Generating Process """

    def __init__(self, features, response, cov_matrix, corr_matrix, gen_details):

        # check params
        self.validate_results(features, response, cov_matrix, corr_matrix, gen_details)

    def validate_results(self, features, response, cov_matrix, corr_matrix, gen_details):

        if features is None:
            raise ValueError("Result 'features' is not valid")
        else:
            self.features = features

        if response is None:
            raise ValueError("Result 'response' is not valid")
        else:
            self.response = response

        if cov_matrix is None:
            raise ValueError("Result 'cov_matrix' is not valid")
        else:
            self.cov_matrix = cov_matrix

        if corr_matrix is None:
            raise ValueError("Result 'corr_matrix' is not valid")
        else:
            self.corr_matrix = corr_matrix
        
        if gen_details is None:
            raise ValueError("Result 'gen_details' is not valid")
        else:
            self.gen_details = gen_details