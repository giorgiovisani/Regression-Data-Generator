import unittest
from unittest import TestCase
import pandas as pd
import numpy as np
import numpy.testing as nt
import sys
import os
from pandas.testing import assert_frame_equal, assert_series_equal

sys.path.append("../../GENERATOR/")

from model.simple_access.functions import Generator
from model.simple_access.functions import IOfunctions

#lr
from model.implementation_lr.lr_params import LR_Params
from model.implementation_lr.lr_generator import LR_Generator, LR_Chunk_Generator
#log
from model.implementation_log.log_params import Log_Params
from model.implementation_log.log_generator import Log_Generator, Log_Chunk_Generator

from model.utilities.csv_saver import Csv_Results_Saver

class LR_Test(TestCase):

    def setUp(self):
        """
        I test vengono eseguiti in ordine alfabetico, perciò metto un numero
        """
        
    def test_01_generated_dataset_without_opt(self):

        #TESTING GENERATED DATASET WITHOUT OPTIMIZATION (float 64)


        # example_dataset è il dataset di esempio, dovrà essere confrontato con quello generato
        # example_dataset è stato creato da un LR_Regressor con i seguenti parametri:

        """
            Si fa il test su un dataset di esempio, generato con determinati parametri
            e vedere se i risultati tornano utilizzando lo stesso seed
        """
        
        example_dataset = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "dataset_float64.csv"), header=None)
        example_betas = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "betas_float64.csv"), header=None)
        example_corr_matrix = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "corr_matrix_float64.csv"), header=None)
        example_cov_matrix = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "cov_matrix_float64.csv"), header=None)

        # must convert ample from DataFrame to Series for checking equality
        example_betas = example_betas.iloc[:, 0]
        
        elements = IOfunctions.get_input(os.path.join("test_generator_lr", "input_elements_float64.json"))
        results = Generator.by_input(elements)
        
        dataset = pd.concat([results.response, results.features], axis=1, sort=False)
        
        y1 = example_dataset.iloc[:, 0].values
        y2 = dataset.iloc[:, 0].values

        # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
        dataset.columns = range(dataset.shape[1])
        example_dataset.columns = range(example_dataset.shape[1])

        example_betas.columns = range(example_betas.shape[0])
        elements.params.betas.columns = range(elements.params.betas.shape[0])
        
        example_corr_matrix.columns = range(example_corr_matrix.shape[1])
        results.corr_matrix.columns = range(results.corr_matrix.shape[1])

        example_cov_matrix.columns = range(example_cov_matrix.shape[1])
        results.cov_matrix.columns = range(results.cov_matrix.shape[1])

        # se decommento questo il test fallisce
        # dataset.at[0, 2] = 304 


        """
            A causa di alcune cifre dopo la virgola, alcuni numeri potrebbero non risultare uguali.
            Per questo uso la funione np.allclose() che controlla che i numeri siano 'circa' uguali
            Per vederlo chiaramente, basta decommentare questo pezzo di codice
            print(arr1) # array of dataset 1
            print(arr2) # array of dataset 2

            i = 0
            while i < arr1.size:
                if arr1[i] == arr2[i]:
                    print("TRUE")
                else:
                    print("FALSE")
                 i += 1
        """
        
        self.assertTrue(np.allclose(y1, y2))
        assert_frame_equal(dataset, example_dataset)
        assert_series_equal(example_betas, elements.params.betas, check_names=False)
        assert_frame_equal(example_corr_matrix, results.corr_matrix)
        assert_frame_equal(example_cov_matrix, results.cov_matrix)
        
        
    def test_02_generated_dataset_with_opt_level_1(self):

        #TESTING GENERATED DATASET WITH OPTIMIZATION LEVEL 1 (FLOAT 32) 


        # example_dataset è il dataset di esempio, dovrà essere confrontato con quello generato
        # example_dataset è stato creato da un LR_Regressor con i seguenti parametri:

        """
            Si fa il test su un dataset di esempio, generato con determinati parametri
            e vedere se i risultati tornano utilizzando lo stesso seed
        """
        
        example_dataset = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "dataset_float32.csv"), header=None)
        example_betas = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "betas_float32.csv"), header=None)
        example_corr_matrix = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "corr_matrix_float32.csv"), header=None)
        example_cov_matrix = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "cov_matrix_float32.csv"), header=None)
        
        # forcing to convert dataset to float32 (.astype() doesn't work)
        floats = example_dataset.select_dtypes(include=['float64']).columns.tolist()
        example_dataset[floats] = example_dataset[floats].apply(pd.to_numeric, downcast='float')

        # must convert ample from DataFrame to Series for checking equality
        example_betas = example_betas.iloc[:, 0]      

        elements = IOfunctions.get_input(os.path.join("test_generator_lr", "input_elements_float32.json"))        
        results = Generator.by_input(elements)
        
        dataset = pd.concat([results.response, results.features], axis=1, sort=False)
        
        y1 = example_dataset.iloc[:, 0].values
        y2 = dataset.iloc[:, 0].values

        # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
        dataset.columns = range(dataset.shape[1])
        example_dataset.columns = range(example_dataset.shape[1])

        example_betas.columns = range(example_betas.shape[0])
        elements.params.betas.columns = range(elements.params.betas.shape[0])
        
        example_corr_matrix.columns = range(example_corr_matrix.shape[1])
        results.corr_matrix.columns = range(results.corr_matrix.shape[1])

        example_cov_matrix.columns = range(example_cov_matrix.shape[1])
        results.cov_matrix.columns = range(results.cov_matrix.shape[1])

        # se decommento questo il test fallisce
        # dataset.at[0, 2] = 304 


        """
            A causa di alcune cifre dopo la virgola, alcuni numeri potrebbero non risultare uguali.
            Per questo uso la funione np.allclose() che controlla che i numeri siano 'circa' uguali
            Per vederlo chiaramente, basta decommentare questo pezzo di codice
            print(arr1) # array of dataset 1
            print(arr2) # array of dataset 2

            i = 0
            while i < arr1.size:
                if arr1[i] == arr2[i]:
                    print("TRUE")
                else:
                    print("FALSE")
                 i += 1
        """
        
        self.assertTrue(np.allclose(y1, y2))
        assert_frame_equal(dataset, example_dataset)
        assert_series_equal(example_betas, elements.params.betas, check_names=False)
        assert_frame_equal(example_corr_matrix, results.corr_matrix)
        assert_frame_equal(example_cov_matrix, results.cov_matrix)
        
    def test_03_generated_dataset_with_opt_level_2(self):

        #TESTING GENERATED DATASET WITH OPTIMIZATION LEVEL 2 (FLOAT 16)


        # example_dataset è il dataset di esempio, dovrà essere confrontato con quello generato
        # example_dataset è stato creato da un LR_Regressor con i seguenti parametri:

        """
            Si fa il test su un dataset di esempio, generato con determinati parametri
            e vedere se i risultati tornano utilizzando lo stesso seed
        """
        
        example_dataset = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "dataset_float16.csv"), header=None)
        example_betas = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "betas_float16.csv"), header=None)
        example_corr_matrix = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "corr_matrix_float16.csv"), header=None)
        example_cov_matrix = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr", "cov_matrix_float16.csv"), header=None)
        
        # convert dataset to float16
        example_dataset = pd.DataFrame(example_dataset.values.astype(dtype='float16', copy=False))

        # must convert ample from DataFrame to Series for checking equality
        example_betas = example_betas.iloc[:, 0]

        elements = IOfunctions.get_input(os.path.join("test_generator_lr", "input_elements_float16.json"))
        results = Generator.by_input(elements)
        
        dataset = pd.concat([results.response, results.features], axis=1, sort=False)
        
        y1 = example_dataset.iloc[:, 0].values
        y2 = dataset.iloc[:, 0].values

        # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
        dataset.columns = range(dataset.shape[1])
        example_dataset.columns = range(example_dataset.shape[1])

        example_betas.columns = range(example_betas.shape[0])
        elements.params.betas.columns = range(elements.params.betas.shape[0])
        
        example_corr_matrix.columns = range(example_corr_matrix.shape[1])
        results.corr_matrix.columns = range(results.corr_matrix.shape[1])

        example_cov_matrix.columns = range(example_cov_matrix.shape[1])
        results.cov_matrix.columns = range(results.cov_matrix.shape[1])

        # se decommento questo il test fallisce
        # dataset.at[0, 2] = 304 

        self.assertTrue(np.allclose(y1, y2))
        assert_frame_equal(dataset, example_dataset)
        assert_series_equal(example_betas, elements.params.betas, check_names=False)
        assert_frame_equal(example_corr_matrix, results.corr_matrix)
        assert_frame_equal(example_cov_matrix, results.cov_matrix)


    def test_04_generated_dataset_chunk(self):

        #TESTING CHUNK BY CHUNK GENERATED DATASET WITHOUT OPTIMIZATION 

        example_dataset = pd.read_csv(os.path.join(os.getcwd(), "input", "test_generator_lr_chunk", "dataset_chunk.csv"), header=None)
        
        elements = IOfunctions.get_input(os.path.join("test_generator_lr_chunk", "input_elements_chunk.json"))
        results = Generator.by_input(elements)
        #I read the output of the generation
        dataset = pd.read_csv(os.path.join(os.getcwd(), "output", "test_generator_lr_chunk", "dataset.csv"), header=None)

      

        y1 = example_dataset.iloc[:, 0].values
        y2 = dataset.iloc[:, 0].values

        # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
        dataset.columns = range(dataset.shape[1])
        example_dataset.columns = range(example_dataset.shape[1])
        
        self.assertTrue(np.allclose(y1, y2))
        self.assertTrue(np.allclose(dataset, example_dataset)) # senza atol va bene perche in questo caso è stato generato con float64
    
if __name__ == '__main__':
    unittest.main()